@ECHO OFF

pushd %~dp0

REM Command file for UML diagram generation

if "%1" == "svg" goto svg
if "%1" == "png" goto png

java -jar plantuml.jar -h
goto end

:svg
java -jar plantuml.jar -tsvg -o "../out" "diagrams/src"

:png
java -jar plantuml.jar -tpng -o "../out" "diagrams/src"

:end
popd