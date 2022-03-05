#include <iostream>

#include <string>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
// using namespace cv;

const char* keys = {
    "{help h usage ? | | print this message}"
    "{@video | | Video file, if not defined try to use webcamera}s"
};


const int Cv_GUI_NORMAL= 0x10;



int main (int argc, const char ** argv)
{
    cv::CommandLineParser parser(argc,argv,keys);
    parser.about("MY ABOUT");

    if(parser.has("help")){
        parser.printMessage();
        return 0;
    }
    
    
    cv::String vid_file = parser.get<cv::String>(0);

    if ( !parser.check() ){
        std::cout<<"THIS E"<<std::endl;
        parser.printErrors();
    }
    cv::VideoCapture cap;
    cap.open(vid_file);
    if(!cap.isOpened())
    {
        return -1;
    }
    cv::FileStorage fs("test.yml",cv::FileStorage::WRITE);
    int fps = cap.get(cv::CAP_PROP_FPS);

    std::cout<<"FPS "<<fps<<std::endl;

    fs<<"FPS "<<fps;

    cv::Mat me = cv::Mat::eye(4,4,CV_32F);
    cv::Mat mo = cv::Mat::ones(4,4,CV_32F);

    cv::Mat res = (mo+1).mul(me+2);
    fs<<"RES"<<res;

    fs.release();


    cv::FileStorage fs2 = cv::FileStorage("test.yml",cv::FileStorage::READ);

    cv::Mat rr;

    fs2["RES"] >> rr;

    std::cout<<rr<<std::endl;

    fs2.release();

    cv::namedWindow("VID");
    cv::Mat frame;
    for(;;)
    {
        cap>>frame;
        if(frame.empty())
        {
            break;
        }
        cv::imshow("VID",frame);
        if(cv::waitKey(15)>0)
        {
            break;
        }
    }

    cap.release();
    return 0;
}



