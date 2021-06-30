# PythonDataRetrieval
Just a library of python data retrieval stuff to be reused.

## Installation
 `pip install git+https://github.com/Deep-Stonks-Group/PythonDataProcessing.git `

### If you are on windows follow these steps:

Download [ta-lib-0.4.0-msvc.zip](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip)
and unzip to ``C:\ta-lib``.

> This is a 32-bit binary release.  If you want to use 64-bit Python, you will
> need to build a 64-bit version of the library. Some unofficial (**and
> unsupported**) instructions for building on 64-bit Windows 10, here for
> reference:
>
> 1. Download and Unzip ``ta-lib-0.4.0-msvc.zip``
> 2. Move the Unzipped Folder ``ta-lib`` to ``C:\``
> 3. Download and Install Visual Studio Community 2015
>    * Remember to Select ``[Visual C++]`` Feature
> 4. Build TA-Lib Library
>    * From Windows Start Menu, Start ``[VS2015 x64 Native Tools Command
>      Prompt]``
>    * Move to ``C:\ta-lib\c\make\cdr\win32\msvc``
>    * Build the Library ``nmake``

Got instructions from: https://github.com/mrjbq7/ta-lib#dependencies

You might also try these unofficial windows binaries for both 32-bit and
64-bit:

https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
