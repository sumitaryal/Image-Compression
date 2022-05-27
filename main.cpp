#include <iostream>
#include <opencv2/opencv.hpp>
#include "compress.cpp"
int main(int argc, char** argv)
{

	Compressor c;
  // char* str = "/Users/sumitaryal/Desktop/SUMIT/ImageCompression/random.bmp" ;
	// c.compress(str);
  char* compressedfile = "/Users/sumitaryal/Desktop/SUMIT/ImageCompression/random.compressed";
  c.decompress(compressedfile);
	return 0;
}