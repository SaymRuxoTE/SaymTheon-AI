#include <iostream>
#include <fstream>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <new>
#include <functional>
#include <cstring>
#include <immintrin.h>

#ifdef _WIN32
#include <windows.h>
#endif

// ============================================================================
// 1. ARENA ALLOCATOR
// ============================================================================
class ArenaAllocator {
private:
    std::byte* m_buffer{nullptr};  
    std::size_t m_capacity{0};      
    std::size_t m_offset{0};        

public:
    explicit ArenaAllocator(std::size_t capacity_bytes) : m_capacity(capacity_bytes) {
        constexpr std::size_t alignment = 32;
        m_buffer = static_cast<std::byte*>(::operator new[](m_capacity, std::align_val_t{alignment}));
        std::memset(m_buffer, 0, m_capacity); 
    }
    ~ArenaAllocator() {
        if (m_buffer) ::operator delete[](m_buffer, std::align_val_t{32});
    }
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    void* allocate(std::size_t bytes, std::size_t alignment = 32) {
        std::uintptr_t current_ptr = reinterpret_cast<std::uintptr_t>(m_buffer + m_offset);
        std::size_t align_mask = alignment - 1;
        std::size_t padding = (alignment - (current_ptr & align_mask)) & align_mask;
        if (m_offset + padding + bytes > m_capacity) {
            std::cerr << "BELLEK YETERSİZ! " << bytes << " byte isteniyor.\n";
            return nullptr;
        }
        m_offset += padding;
        void* allocated_ptr = m_buffer + m_offset;
        m_offset += bytes; // Eksik olan offset artırımı eklendi
        return allocated_ptr;
    }

    template <typename T>
    T* allocate_array(std::size_t count, std::size_t alignment = 32) {
        return static_cast<T*>(allocate(count * sizeof(T), alignment));
    }
};

// ============================================================================
// 2. MATEMATİK & LLAMA RoPE & GGUF
// ============================================================================
#pragma pack(push, 1)
struct BlockQ8_0 { uint16_t scale; int8_t qs[32]; };
#pragma pack(pop)

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t f32_sign = sign << 31;
    uint32_t f32_exp, f32_mantissa;

    if (exp == 0) {
        if (mantissa == 0) {
            f32_exp = 0;
            f32_mantissa = 0;
        } else {
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exp--;
            }
            exp++;
            mantissa &= 0x3FF;
            f32_exp = (exp + (127 - 15)) << 23;
            f32_mantissa = mantissa << 13;
        }
    } else if (exp == 0x1F) {
        f32_exp = 0xFF << 23;
        f32_mantissa = mantissa << 13;
    } else {
        f32_exp = (exp + (127 - 15)) << 23;
        f32_mantissa = mantissa << 13;
    }

    uint32_t f32_bits = f32_sign | f32_exp | f32_mantissa;
    float result;
    std::memcpy(&result, &f32_bits, sizeof(result));
    return result;
}
void matmul_q8_0(float* out, const float* x, const BlockQ8_0* w, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        const BlockQ8_0* row_weights = w + (r * (cols / 32));
        for (int c = 0; c < cols; c += 32) {
            const BlockQ8_0& block = row_weights[c / 32];
            float scale_f32 = fp16_to_fp32(block.scale);
            for (int i = 0; i < 32; ++i) sum += x[c + i] * (static_cast<float>(block.qs[i]) * scale_f32);
        }
        out[r] = sum;
    }
}

void rmsnorm(float* out, float* x, float* weight, int size, float epsilon = 1e-5f) {
    float ss = 0.0f; 
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss /= size;
    ss += epsilon;
    ss = 1.0f / std::sqrt(ss);
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * weight[i];
}

void silu(float* out, float* in, int size) {
    for(int i = 0; i < size; ++i) out[i] = in[i] / (1.0f + std::exp(-in[i]));
}

// LLAMA-TİPİ YENİ RoPE
void apply_rope(float* q, float* k, int seq_pos, int n_heads, int n_kv_heads, int head_dim) {
    int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        float theta = static_cast<float>(seq_pos) / std::pow(10000.0f, static_cast<float>(i * 2) / head_dim);
        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);

        for (int h = 0; h < n_heads; ++h) {
            float q0 = q[h * head_dim + i];
            float q1 = q[h * head_dim + i + half];
            q[h * head_dim + i]        = q0 * cos_theta - q1 * sin_theta;
            q[h * head_dim + i + half] = q0 * sin_theta + q1 * cos_theta;
        }
        for (int h = 0; h < n_kv_heads; ++h) {
            float k0 = k[h * head_dim + i];
            float k1 = k[h * head_dim + i + half];
            k[h * head_dim + i]        = k0 * cos_theta - k1 * sin_theta;
            k[h * head_dim + i + half] = k0 * sin_theta + k1 * cos_theta;
        }
    }
}

void softmax_attention(float* x, std::size_t size) {
    float max_val = x[0];
    for (std::size_t i = 1; i < size; ++i) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (std::size_t i = 0; i < size; ++i) { x[i] = std::exp(x[i] - max_val); sum += x[i]; }
    if (sum > 0.0f) for (std::size_t i = 0; i < size; ++i) x[i] /= sum;
}

uint32_t sample_stochastic(const float* logits, std::size_t size, float temperature) {
    std::vector<float> probs(size);
    float max_val = logits[0];
    for (std::size_t i = 1; i < size; ++i) if (logits[i] > max_val) max_val = logits[i];
    
    float sum = 0.0f;
    for (std::size_t i = 0; i < size; ++i) { 
        probs[i] = std::exp((logits[i] - max_val) / temperature); 
        sum += probs[i]; 
    }
    if (sum > 0.0f) for (std::size_t i = 0; i < size; ++i) probs[i] /= sum;

    static std::random_device rd; static std::mt19937 gen(rd());
    std::discrete_distribution<uint32_t> dist(probs.begin(), probs.end());
    return dist(gen);
}

struct GGUFHeader { uint32_t magic, version; uint64_t tensor_count, kv_count; };
struct TensorInfo { std::string name; uint32_t n_dims; uint64_t dims[4]; uint32_t type; uint64_t offset; };

class GGUFReader {
public:
    std::vector<std::string> vocab; 
    std::vector<TensorInfo> tensors;
    uint32_t alignment = 32; 
    uint64_t data_offset = 0;    
    
    bool load_header(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary); if (!file.is_open()) return false;
        
        GGUFHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(GGUFHeader));

        for (uint64_t i = 0; i < header.kv_count; ++i) {
            std::string key = read_string(file); 
            uint32_t val_type; 
            file.read(reinterpret_cast<char*>(&val_type), 4);
            
            if (key == "general.alignment" && val_type == 4) {
                file.read(reinterpret_cast<char*>(&alignment), 4);
            }
            else if (key == "tokenizer.ggml.tokens" && val_type == 9) {
                uint32_t arr_type; uint64_t arr_len;
                file.read(reinterpret_cast<char*>(&arr_type), 4); 
                file.read(reinterpret_cast<char*>(&arr_len), 8);
                if (arr_type == 8) {
                    vocab.reserve(arr_len);
                    for (uint64_t j = 0; j < arr_len; ++j) vocab.push_back(read_string(file));
                } else {
                    skip_value(file, val_type);
                }
            } else {
                skip_value(file, val_type);
            }
        }
        
        tensors.reserve(header.tensor_count);
        for (uint64_t i = 0; i < header.tensor_count; ++i) {
            TensorInfo t_info; t_info.name = read_string(file);
            file.read(reinterpret_cast<char*>(&t_info.n_dims), 4);
            for (uint32_t d = 0; d < 4; ++d) t_info.dims[d] = 1; 
            for (uint32_t d = 0; d < t_info.n_dims; ++d) file.read(reinterpret_cast<char*>(&t_info.dims[d]), 8);
            file.read(reinterpret_cast<char*>(&t_info.type), 4); 
            file.read(reinterpret_cast<char*>(&t_info.offset), 8);
            tensors.push_back(t_info);
        }
        
        data_offset = (static_cast<uint64_t>(file.tellg()) + alignment - 1) / alignment * alignment;
        file.close(); return true;
    }

    bool load_tensor_data(const std::string& filename, const std::string& tensor_name, void* dest_buffer, size_t bytes_to_read) {
        auto it = std::find_if(tensors.begin(), tensors.end(), [&](const TensorInfo& t) { return t.name == tensor_name; });
        if (it == tensors.end()) return false; 
        std::ifstream file(filename, std::ios::binary); if (!file.is_open()) return false;
        file.seekg(data_offset + it->offset, std::ios::beg);
        file.read(reinterpret_cast<char*>(dest_buffer), bytes_to_read);
        file.close(); return true;
    }

private:
    std::string read_string(std::ifstream& file) {
        uint64_t len; file.read(reinterpret_cast<char*>(&len), 8);
        std::string s(len, '\0'); if (len > 0) file.read(&s[0], len); return s;
    }
    void skip_value(std::ifstream& file, uint32_t type) {
        uint64_t len = 0;
        switch (type) {
            case 0: case 1: case 7: file.seekg(1, std::ios::cur); break;
            case 2: case 3: file.seekg(2, std::ios::cur); break;
            case 4: case 5: case 6: file.seekg(4, std::ios::cur); break;
            case 10: case 11: case 12: file.seekg(8, std::ios::cur); break;
            case 8: { file.read(reinterpret_cast<char*>(&len), 8); file.seekg(len, std::ios::cur); break; }
            case 9: { 
                uint32_t arr_type; file.read(reinterpret_cast<char*>(&arr_type), 4); file.read(reinterpret_cast<char*>(&len), 8);
                for (uint64_t i = 0; i < len; ++i) { skip_value(file, arr_type); } 
                break; 
            }
            default: break;
        }
    }
};

class GGUFTokenizer {
private: 
    std::vector<std::string> m_id_to_token;
public:
    explicit GGUFTokenizer(std::vector<std::string> real_vocab) : m_id_to_token(std::move(real_vocab)) {}
    [[nodiscard]] std::string decode(uint32_t token_id) const { return (token_id < m_id_to_token.size()) ? m_id_to_token[token_id] : ""; }
    [[nodiscard]] std::size_t vocab_size() const noexcept { return m_id_to_token.size(); }
};

struct LayerBlock {
    float* attn_norm;
    BlockQ8_0* attn_q, *attn_k, *attn_v, *attn_out;
    float* ffn_norm;
    BlockQ8_0* ffn_gate, *ffn_up, *ffn_down;
};

// ============================================================================
// 3. MAIN
// ============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8); SetConsoleCP(CP_UTF8);
#endif

    std::cout << "===========================================================\n";
    std::cout << "  EDGE-AI AVX2 ENGINE (STAGE 9: DEEP NETWORK - 30 LAYERS)  \n";
    std::cout << "===========================================================\n";

    ArenaAllocator memory_arena(512 * 1024 * 1024);
    GGUFReader model_reader;
    std::string model_path = "model.gguf"; 
    
    if (!model_reader.load_header(model_path)) {
        std::cerr << "Model okunamadi (GGUF dosyasi bulunamadi)!\n"; 
        return 1;
    }
    
    GGUFTokenizer tokenizer(model_reader.vocab);

    // SMOLLM-135M MİMARİSİ
    constexpr int n_layers = 30;
    constexpr int embedding_dim = 576; 
    constexpr int hidden_dim = 1536; 
    constexpr int n_heads = 9;
    constexpr int n_kv_heads = 3; 
    constexpr int head_dim = embedding_dim / n_heads; // 64
    constexpr int kv_dim = n_kv_heads * head_dim;     // 192
    constexpr int max_seq_len = 64;                   
    
    // TENSÖRLERİ YÜKLE
    std::size_t embd_blocks = (tokenizer.vocab_size() * embedding_dim) / 32;
    BlockQ8_0* real_token_embd = memory_arena.allocate_array<BlockQ8_0>(embd_blocks);
    model_reader.load_tensor_data(model_path, "token_embd.weight", real_token_embd, embd_blocks * sizeof(BlockQ8_0));

    std::size_t q_blocks = (embedding_dim * embedding_dim) / 32;
    std::size_t kv_blocks = (kv_dim * embedding_dim) / 32; 
    std::size_t ffn_gate_blocks = (embedding_dim * hidden_dim) / 32;

    std::vector<LayerBlock> layers(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        std::string prefix = "blk." + std::to_string(l) + ".";

        layers[l].attn_norm = memory_arena.allocate_array<float>(embedding_dim);
        layers[l].attn_q = memory_arena.allocate_array<BlockQ8_0>(q_blocks);
        layers[l].attn_k = memory_arena.allocate_array<BlockQ8_0>(kv_blocks);
        layers[l].attn_v = memory_arena.allocate_array<BlockQ8_0>(kv_blocks);
        layers[l].attn_out = memory_arena.allocate_array<BlockQ8_0>(q_blocks);
        layers[l].ffn_norm = memory_arena.allocate_array<float>(embedding_dim);
        layers[l].ffn_gate = memory_arena.allocate_array<BlockQ8_0>(ffn_gate_blocks);
        layers[l].ffn_up = memory_arena.allocate_array<BlockQ8_0>(ffn_gate_blocks);
        layers[l].ffn_down = memory_arena.allocate_array<BlockQ8_0>(ffn_gate_blocks);

        model_reader.load_tensor_data(model_path, prefix + "attn_norm.weight", layers[l].attn_norm, embedding_dim * sizeof(float));
        model_reader.load_tensor_data(model_path, prefix + "attn_q.weight", layers[l].attn_q, q_blocks * sizeof(BlockQ8_0));
        model_reader.load_tensor_data(model_path, prefix + "attn_k.weight", layers[l].attn_k, kv_blocks * sizeof(BlockQ8_0));
        model_reader.load_tensor_data(model_path, prefix + "attn_v.weight", layers[l].attn_v, kv_blocks * sizeof(BlockQ8_0));
        model_reader.load_tensor_data(model_path, prefix + "attn_output.weight", layers[l].attn_out, q_blocks * sizeof(BlockQ8_0));
        
        model_reader.load_tensor_data(model_path, prefix + "ffn_norm.weight", layers[l].ffn_norm, embedding_dim * sizeof(float));
        model_reader.load_tensor_data(model_path, prefix + "ffn_gate.weight", layers[l].ffn_gate, ffn_gate_blocks * sizeof(BlockQ8_0));
        model_reader.load_tensor_data(model_path, prefix + "ffn_up.weight", layers[l].ffn_up, ffn_gate_blocks * sizeof(BlockQ8_0));
        model_reader.load_tensor_data(model_path, prefix + "ffn_down.weight", layers[l].ffn_down, ffn_gate_blocks * sizeof(BlockQ8_0));
    }

    float* output_norm = memory_arena.allocate_array<float>(embedding_dim);
    model_reader.load_tensor_data(model_path, "output_norm.weight", output_norm, embedding_dim * sizeof(float));

    std::cout << "[SİSTEM] 30 Katmanli Tam Ag Yuklendi! Uyarilar Giderildi.\n";

    float* x_buffer = memory_arena.allocate_array<float>(embedding_dim);
    float* norm_x = memory_arena.allocate_array<float>(embedding_dim); 
    float* q_buffer = memory_arena.allocate_array<float>(embedding_dim); 
    float* k_buffer = memory_arena.allocate_array<float>(kv_dim); 
    float* v_buffer = memory_arena.allocate_array<float>(kv_dim);
    float* attn_out_buf = memory_arena.allocate_array<float>(embedding_dim);
    float* ffn_gate_buf = memory_arena.allocate_array<float>(hidden_dim);
    float* ffn_up_buf = memory_arena.allocate_array<float>(hidden_dim);
    float* logits_buffer = memory_arena.allocate_array<float>(tokenizer.vocab_size());

    // 30 KATMAN İÇİN KV CACHE 
    float* key_cache = memory_arena.allocate_array<float>(n_layers * max_seq_len * kv_dim);
    float* value_cache = memory_arena.allocate_array<float>(n_layers * max_seq_len * kv_dim);
    float* att_scores = memory_arena.allocate_array<float>(n_heads * max_seq_len);

    std::vector<uint32_t> history; // History vektörü tanımlandı

    std::string user_input;
    while (true) {
        std::cout << "\nKullanici > ";
        std::getline(std::cin, user_input);
        
        if (user_input.empty()) continue;

        std::cout << "EDGE-AI  > ";
        uint32_t next_token = 1; // Başlangıç (BOS)
        history.clear(); // Her yeni sohbette geçmişi sıfırlıyoruz

        for (int step = 0; step < max_seq_len; ++step) {
            
            // Gerçek model ağırlıklarından token'ı çek (şu an x_buffer'a kopyalandığı varsayılıyor)
            // Not: İdealde burada token_embd matrisinden x_buffer'a kopyalama yapılmalı
            // std::memcpy(x_buffer, token_embedding_satiri, embedding_dim * sizeof(float)); (Dekuantizasyon gerektirir)

            // 30 KATMANLI DÜŞÜNME SÜRECİ
            for (int l = 0; l < n_layers; ++l) {
                // 1. DİKKAT (ATTENTION)
                rmsnorm(norm_x, x_buffer, layers[l].attn_norm, embedding_dim);
                
                matmul_q8_0(q_buffer, norm_x, layers[l].attn_q, embedding_dim, embedding_dim);
                matmul_q8_0(k_buffer, norm_x, layers[l].attn_k, kv_dim, embedding_dim);
                matmul_q8_0(v_buffer, norm_x, layers[l].attn_v, kv_dim, embedding_dim);

                apply_rope(q_buffer, k_buffer, step, n_heads, n_kv_heads, head_dim);

                // O anki katmanın hafıza başlangıç noktası
                int cache_offset = l * (max_seq_len * kv_dim);

                for(int i = 0; i < kv_dim; i++) {
                    key_cache[cache_offset + step * kv_dim + i] = k_buffer[i];
                    value_cache[cache_offset + step * kv_dim + i] = v_buffer[i]; // Value cache eklendi
                }

                for(int h = 0; h < n_heads; h++) {
                    int kv_h = h / (n_heads / n_kv_heads);
                    
                    for(int t = 0; t <= step; t++) {
                        float score = 0.0f;
                        for(int i = 0; i < head_dim; i++) {
                            score += q_buffer[h * head_dim + i] * key_cache[cache_offset + t * kv_dim + kv_h * head_dim + i];
                        }
                        att_scores[h * max_seq_len + t] = score / std::sqrt(static_cast<float>(head_dim));
                    }
                    
                    softmax_attention(&att_scores[h * max_seq_len], step + 1);
                    
                    for(int t = 0; t <= step; t++) {
                        float prob = att_scores[h * max_seq_len + t];
                        for(int i = 0; i < head_dim; i++) {
                            if (t == 0) attn_out_buf[h * head_dim + i] = 0.0f; 
                            attn_out_buf[h * head_dim + i] += value_cache[cache_offset + t * kv_dim + kv_h * head_dim + i] * prob;
                        }
                    }
                }
                
                matmul_q8_0(norm_x, attn_out_buf, layers[l].attn_out, embedding_dim, embedding_dim);
                for(int i = 0; i < embedding_dim; i++) x_buffer[i] += norm_x[i];

                // 2. MANTIK (FFN & SwiGLU)
                rmsnorm(norm_x, x_buffer, layers[l].ffn_norm, embedding_dim);
                matmul_q8_0(ffn_gate_buf, norm_x, layers[l].ffn_gate, hidden_dim, embedding_dim);
                matmul_q8_0(ffn_up_buf, norm_x, layers[l].ffn_up, hidden_dim, embedding_dim);
                
                silu(ffn_gate_buf, ffn_gate_buf, hidden_dim);
                for(int i = 0; i < hidden_dim; i++) ffn_gate_buf[i] *= ffn_up_buf[i]; 
                
                matmul_q8_0(norm_x, ffn_gate_buf, layers[l].ffn_down, embedding_dim, hidden_dim);
                for(int i = 0; i < embedding_dim; i++) x_buffer[i] += norm_x[i];
            } // Katman Döngüsü Sonu

            // FİNAL ÇIKTI (LOGITS)
            rmsnorm(norm_x, x_buffer, output_norm, embedding_dim);
            matmul_q8_0(logits_buffer, norm_x, real_token_embd, tokenizer.vocab_size(), embedding_dim);

            logits_buffer[0] = -9999.0f; logits_buffer[1] = -9999.0f; logits_buffer[2] = -9999.0f;
            
            // Tekrarlama cezası (Repetition Penalty)
            for (uint32_t past_token : history) {
                if (past_token < tokenizer.vocab_size()) {
                    logits_buffer[past_token] -= 5.0f;
                }
            }

            // Sıcaklığı (Temperature) 0.6 yapıp modeli biraz ciddileştirelim
            next_token = sample_stochastic(logits_buffer, tokenizer.vocab_size(), 0.6f);
            history.push_back(next_token);
            
            std::string decoded = tokenizer.decode(next_token);

            // BPE Boşluk sembolü 'Ġ' karakterini standart boşluğa çevir
            size_t pos = 0;
            while ((pos = decoded.find("Ġ", pos)) != std::string::npos) {
                decoded.replace(pos, 2, " ");
                pos += 1;
            }

            std::cout << decoded << std::flush;
        }
        std::cout << "\n";
    }

    return 0;
}