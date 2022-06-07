#include <assert.h>
#include <stdio.h>


// use const to indicate input str
// use NULL to avoid segmentation fault
// use `return address;` to func invocation friendly (user can write 
//          code such as `char* strCopy = strcpy(strCopy, strSrc);`)
char * strcpy( char *strDest, const char *strSrc )
{
    assert( (strDest != NULL) && (strSrc != NULL) );
    char *address = strDest;
    while( (*strDest++ = * strSrc++) != '\0' );
    return address;
}
