

#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

// using namespace std;


int main(int argc, const char** argv)
{
    std::cout<<"IMREAD EXAMPLE"<<std::endl;
    cv::Mat color = cv::imread(argv[1]);
    cv::Mat gray = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    cv::imwrite("img_gray.PNG", gray);

    cv::imshow("COLOR",color);
    cv::imshow("GRAY",gray);

    int row = 100;
    int col = 200;

    cv::Vec3b pixel = color.at<cv::Vec3b>(row,col);

    
    std::cout<<"BGR\t"<<(int) pixel[0]<<"\t"<<(int) pixel[1]<<"\t"<<(int) pixel[2]<<"\t"<<std::endl;


    cv::waitKey(0);

    

    return 0;

}
