@echo off
:anfang
cls
color c
echo THANK YOU !!! 
echo HERE IS A SCRIPT TO CALCULATE PI AFTER FORMULA OF CHUDNOVSKY
pause
:B
cls
color e
date /T
time /T
color a
echo calculate Number [PI] with 6 GB of Ram up to 32 Billion Digits long? 
echo calculate Number Pi with "python" script short version (1.000.000 digits)
echo (64 GB RAM totaly needed - can crash System - use at own risk)
echo open my [HOMEPAGE] ?
echo [exit]


set /p variable=


if /i %variable% == PI goto :INGE
if /i %variable% == Pi goto :INGE
if /i %variable% == pi goto :INGE
if /i %variable% == someone goto :M
if /i %variable% == Someone goto :M
if /i %variable% == SOMEONE goto :M
if /i %variable% == HOMEPAGE goto :CASU
if /i %variable% == Homepage goto :CASU
if /i %variable% == homepage goto :CASU
if /i %variable% == python goto :python
if /i %variable% == Python goto :python
if /i %variable% == PYTTHON goto :python
if /i %variable% == EXIT goto :Q
if /i %variable% == exit goto :Q
if /i %variable% == Exit goto :Q


goto :anfang

:CASU
@echo off
start https://lino.lol
goto :anfang

:INGE
tpi -T 8 -m 6Gi -d 32G -o pi.txt 32G
echo now writing in file pi.txt?
pause
set zahlen="pi.txt"
type %zahlen% 
echo END OF DATA
pause
goto :anfang

:M
@echo off
y-cruncher.exe
pause
goto :anfang

:python 
@echo off
.\pi-install-python.bat
pause
goto :anfang


:LINO
echo off
cls
color c
echo ERROR - GURU IS MEDITATING
pause
goto :anfang

:Q
@echo off
exit


