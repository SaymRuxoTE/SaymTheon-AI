@echo off
title EDGE-AI Motoru
color 0A

echo [SISTEM] Derleme baslatiliyor...
echo.

:: Eğer src klasörüne girmemiz gerekiyorsa giriyoruz
cd src

:: Kodu derliyoruz
g++ -std=c++20 -O3 -mavx2 -mf16c -mfma -ffast-math -o edge_ai_engine.exe main.cpp

:: Eğer derlemede hata olursa (errorlevel 0'dan farklıysa) işlemi durdur
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Derleme sirasinda bir sorun olustu! Kodu kontrol et.
    pause
    exit /b
)

echo [SISTEM] Derleme basarili! Motor calistiriliyor...
echo ===================================================
echo.

:: Programı çalıştır
edge_ai_engine.exe

:: Program kapanırsa pencere hemen kapanmasın diye bekle
pause