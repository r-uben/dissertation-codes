//
//  PrintMacros.cpp
//  emv-algorithm-c++
//
//  Created by Rubén Fernández Fuertes on 13/7/21.
//

#include <iostream>
#include <fstream>
#include <limits>
using namespace std;

#define COMMA               << ", ";    
#define START_LINE          cout <<
#define END_LINE            << endl;
#define OUTPUT              output <<

#define END_PRINTING(a)     START_LINE (a) END_LINE
#define PRINT(a)            START_LINE (a) COMMA

#define END_WRITING(a)      OUTPUT (a) END_LINE
#define WRITE(a)            OUTPUT fixed << setprecision(10) << (a) COMMA

/// CONSOLE
// RECURSION FOR PRINTING RESULTS IN THE CONSOLE: WE FIRSTLY DECLARE THE "LAST STATE" IN THE
// RECURSION, I.E., WHEN THERE'S ONLY ONE ARGUMENT LEFT TO PRINTING.
// THE SECOND TEMPLATE DOES THE REST.

template<typename T>                    // Type is resolved in compile time
void PRINT_DATA_LINE(T t)
{
    END_PRINTING(t)
}

template<typename T, typename... ARGS>
void PRINT_DATA_LINE(T t, ARGS... args) // Take the first arguments
{
    PRINT(t)
    PRINT_DATA_LINE(args...);           // Recursion with a fewer argument
}

/// CSV
// (SAME PROCEDURE THAN BEFORE)
template<typename T>
void DATA_LINE(ofstream *output, T t)
{
    *END_WRITING(t)
}

template<typename T, typename... ARGS>
void DATA_LINE(ofstream *output, T t, ARGS... args)
{
    *WRITE(t)
    DATA_LINE(output, args...);
}

void OpenCSVFile(ofstream *output, string name, bool over_write){
    // open up a file stream to write data
    if (over_write == true)
        (*output).open("/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/MSc Mathematical Finance - Manchester/subjects/II_semester/MATH60082_computational_finance/c++/assignment2/data/" + name + ".csv");
    if (over_write == false)
        (*output).open("/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/MSc Mathematical Finance - Manchester/subjects/II_semester/MATH60082_computational_finance/c++/assignment2/data/" + name + ".csv", fstream::app);
    // check if the file is opened
    if (!(*output).is_open()){
        PRINT_DATA_LINE(" File not opened");
        // stop here
        throw;
    }
}


