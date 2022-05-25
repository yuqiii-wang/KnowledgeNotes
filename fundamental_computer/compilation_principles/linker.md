# Linker

Reference:
https://docs.oracle.com/cd/E23824_01/html/819-0690/glcdi.html#scrolltoc

* Link-Editor

The link-editor, ld, concatenates and interprets data from one or more input files. These files can be relocatable objects, shared objects, or archive libraries. From these input files, one output file is created. This file is either a relocatable object, dynamic executable, or a shared object. The link-editor is most commonly invoked as part of the compilation environment.

* Runtime Linker

The runtime linker, ld.so.1, processes dynamic executables and shared objects at runtime, binding the executable and shared objects together to create a runnable process.

* Shared Objects

Shared objects are one form of output from the link-edit phase. Shared objects are sometimes referred to as Shared Libraries. Shared objects are importance in creating a powerful, flexible runtime environment.