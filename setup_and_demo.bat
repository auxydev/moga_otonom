@echo off
rem Moga Otonom - Windows: kurulum + demo (cift tiklayarak calistirilabilir)
rem MetaDrive ile: setup_and_demo.bat -MetaDrive
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_and_demo.ps1" %*
pause
