@echo off
REM sync-blog.bat - 博客同步批处理包装器
REM 双击运行或添加到任务计划程序

cd /d "D:\virtualMachine\github\GEOPro-git\GEOPro\idea\worldsense-blog"
powershell -ExecutionPolicy Bypass -File scripts\sync-blog.ps1
