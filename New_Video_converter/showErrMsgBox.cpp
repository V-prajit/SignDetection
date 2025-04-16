#include "showErrMsgBox.h"
#include <stdio.h>

void displayMessageBox(const char *errMessage, FILE * error_outfile) 
{
	fprintf(stderr, "Error: %s\n", errMessage);
	if (error_outfile != NULL && error_outfile != stderr) {
		fprintf(error_outfile, "Error: %s\n", errMessage);
   }
}