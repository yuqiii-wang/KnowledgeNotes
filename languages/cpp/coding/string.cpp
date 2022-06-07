#include <stdio.h>
#include <string.h>

class String
{
public:
    String(const char *str = NULL); // default constructor
    String(const String &other); // copy constructor
    ~ String(void); // destructor
    String & operator =(const String &other); // assignment
private:
    char *m_data; // internal data area
};


String::String(const char *str)
{
    if(str==NULL)
    {
        m_data = new char[1];
        *m_data = '\0';
    }
    else
    {
        int length = strlen(str);
         m_data = new char[length+1];
         strcpy(m_data, str);
    }
}

String::~String(void)
{
    delete [] m_data;
}

String::String(const String &other)
{
    int length = strlen(other.m_data);
    m_data = new char[length+1];    
    strcpy(m_data, other.m_data);
}

String & String::operator =(const String &other)
{
    if(this == &other)  // remember to return itself
        return *this;
    delete [] m_data;   // remember to empty existing resource
    int length = strlen( other.m_data );
    m_data = new char[length+1];   
    strcpy( m_data, other.m_data );
    return *this;
}