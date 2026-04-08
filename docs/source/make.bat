@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	py -m sphinx.cmd.build --version >NUL 2>NUL
	if not errorlevel 1 (
		set SPHINXBUILD=py -m sphinx.cmd.build
		goto sphinx_ready
	)
	python -m sphinx.cmd.build --version >NUL 2>NUL
	if not errorlevel 1 (
		set SPHINXBUILD=python -m sphinx.cmd.build
		goto sphinx_ready
	)
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed ^(e.g. pip install -e ".[docs]"^), then set SPHINXBUILD
	echo.to the full path of sphinx-build, or use Python: py -m sphinx.cmd.build
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

:sphinx_ready
%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.Sphinx command still not available after fallback.
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
