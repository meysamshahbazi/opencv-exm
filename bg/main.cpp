
#include <iostream>
#include <sstream>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"


const char* params
    = "{ help h         |           | Print usage }"
      "{ input          | vtest.avi | Path to a video or a sequence of image }"
      "{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";


int main(int argc,char * argv[] )
{
    cv::CommandLineParser parser(argc,argv,params);
    parser.about("BG");

    if (parser.has("help"))
    {
        //print help information
        parser.printMessage();
    }

    cv::Ptr<cv::BackgroundSubtractor> pbs;
    if(parser.get<std::string>("algo")=="MOG2")
    {
        pbs = cv::createBackgroundSubtractorMOG2();
    }
    else
    {
        pbs = cv::createBackgroundSubtractorKNN();
    }

    cv::VideoCapture vc(argv[1]);

    if (!vc.isOpened())
    {
        //error in opening the video input
        std::cerr << "Unable to open: " << parser.get<cv::String>("input") << std::endl;
        return 0;
    }


    cv::Mat frame,fgmask;
    cv::Mat res;
    cv::Mat temp;
    cv::Mat fgmask2;

    while (1)
    {
        vc>>frame;
        if (frame.empty())
        {
            break;
        }
        pbs->apply(frame,fgmask);

        cv::imshow("Frame", frame);
        // cv::imshow("FG Mask", fgmask);
        
        fgmask.convertTo(fgmask2,CV_8UC3);

        // std::cout<<frame.type()<<std::endl;
        // std::cout<<fgmask2.type()<<std::endl;
        // std::cout<<"---------------------"<<std::endl;
        // std::cout<<CV_8UC3<<std::endl;

        // std::cout<<CV_8UC1<<std::endl;


        // cv::add(frame,fgmask,res);
        
        cv::Mat o = cv::Mat::ones(frame.size[0],frame.size[1],frame.type());
        o = 127*o;
        std::vector<cv::Mat> v;
        v.push_back(fgmask);
        v.push_back(fgmask);
        v.push_back(fgmask);
        cv::merge(v,fgmask2);
        // std::cout<<fgmask2.depth()<<std::endl;
        // cv::Mat o = cv:::Mat::one
        cv::add(frame,fgmask2,res);
        // cv::add(frame,o,res,255-fgmask);

        cv::imshow("res",res);
        //get the input from the keyboard
        int keyboard = cv::waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        
    }
    


    return 0;

}

