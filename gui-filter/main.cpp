
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>


#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

cv::Mat img;

const char* keys =
{
	"{help h usage ? | | print this message}"
    "{@image | | Image to process}"
};






int main(int argc, const char** argv) 
{
    cv::CommandLineParser parser(argc,argv,keys);

    parser.about("some Filter with insha'llah gui");

    if(parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string imgFile = parser.get<std::string>(0);

    img = cv::imread(imgFile);
    std::cout<<img.size()<<std::endl;
    cv::namedWindow("Input");
    cv::imshow("Input", img);

    // SHow Histogram call back 
    std::vector<cv::Mat> bgr;
    cv::split(img,bgr);
    int nb_bin = 256;

    float range[] = {0,256};
    const float* hist_range = {range};

    // std::cout<<<<std::endl;
    cv::Mat b_hist, g_hist, r_hist;

    cv::calcHist(&bgr[0],1,0,cv::Mat(),b_hist,1,&nb_bin,&hist_range);
    cv::calcHist(&bgr[1],1,0,cv::Mat(),g_hist,1,&nb_bin,&hist_range);
    cv::calcHist(&bgr[2],1,0,cv::Mat(),r_hist,1,&nb_bin,&hist_range);

    int width = 512;
    int height = 300;


    cv::Mat hist_img(height,width,CV_8UC3,cv::Scalar(20,20,20));

    cv::normalize(b_hist,b_hist,0,height,cv::NORM_MINMAX);
    cv::normalize(g_hist,g_hist,0,height,cv::NORM_MINMAX);
    cv::normalize(r_hist,r_hist,0,height,cv::NORM_MINMAX);

    int binStep = cvRound((float)width / (float)nb_bin);
	for (int i = 1; i < nb_bin; i++)
	{
		cv::line(hist_img,
			cv::Point(binStep*(i - 1), height - cvRound(b_hist.at<float>(i - 1))),
			cv::Point(binStep*(i), height - cvRound(b_hist.at<float>(i))),
			cv::Scalar(255, 0, 0)
			);
		cv::line(hist_img,
			cv::Point(binStep*(i - 1), height - cvRound(g_hist.at<float>(i - 1))),
			cv::Point(binStep*(i), height - cvRound(g_hist.at<float>(i))),
			cv::Scalar(0, 255, 0)
			);
		cv::line(hist_img,
			cv::Point(binStep*(i - 1), height - cvRound(r_hist.at<float>(i - 1))),
			cv::Point(binStep*(i), height - cvRound(r_hist.at<float>(i))),
			cv::Scalar(0, 0, 255)
			);
	}

	cv::imshow("Histogram", hist_img);
    // END of hisogram 
    // ----------------------------------------------------------------------------
    // Equlize hisogram 

    cv::Mat res;
    cv::Mat ycrcb;

    cv::cvtColor(img,ycrcb,cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> chs;
    cv::split(ycrcb,chs);
    cv::equalizeHist(chs[0],chs[0]);
    cv::merge(chs,ycrcb);
    cv::cvtColor(ycrcb,res,cv::COLOR_YCrCb2BGR);
    cv::imshow("EQZD",res);
    
    // ----------------------------------------------------------------------------

    cv::Mat lomo;
    const double E = std::exp(1.0);
    cv::Mat lut(1, 256, CV_8UC1);

    for(int i = 0; i <256; i++)
    {
        float x = (float)i/256;
        // lut.at<uchar>(i) = cvRound( 256*(1/(1+pow(E,-((x-0.5)/0.1) ))));
        lut.at<uchar>(i) = cvRound(256 * (1 / (1 + pow(E, -((x - 0.5) / 0.1)))));
    }
    

    img = cv::imread(imgFile);
    std::vector<cv::Mat> bgr1;
    cv::split(img,bgr1);
    std::cout<<"BGR1: "<<bgr1[2].size<<std::endl;

    cv::LUT(bgr1[2],lut,bgr1[2]);
    std::cout<<"BGR1: "<<bgr1[2].size<<std::endl;
    cv::merge(bgr1,lomo);
    
    cv::Mat halo(img.rows,img.cols, CV_32FC3,cv::Scalar(0.3,0.3,0.3) );

    cv::circle(halo,cv::Point(img.cols/2,img.rows/2),img.cols/3,cv::Scalar(1, 1, 1),-1);

    cv::blur(halo, halo, cv::Size(img.cols/3, img.rows/3 ) );


    cv::Mat lomof;
    lomo.convertTo(lomof,CV_32FC3);

    cv::multiply(lomof, halo, lomof);
    
    lomof.convertTo(lomo,CV_8UC3);

    cv::imshow("LOM", lomo);
    // ----------------------------------------------------------------------------

    cv::Mat med_img;
    cv::medianBlur(img,med_img,7);
    cv::Mat canny_img;
    cv::Canny(med_img,canny_img, 50, 150);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(2,2));


    cv::dilate(canny_img, canny_img,kernel);

    canny_img = canny_img/256;
    canny_img = 1-canny_img;

    cv::Mat canny_imgf ;
    canny_img.convertTo(canny_imgf,CV_32FC3);

    cv::blur(canny_imgf,canny_imgf,cv::Size(2,2));

    cv::Mat imgBF;

    cv::bilateralFilter(img,imgBF,9,150.0,150.0);
    cv::imshow("imgBF",imgBF);

    cv::Mat cartoon = imgBF/25;

    cartoon = cartoon*25;

    cv::Mat canny_img3c;
    cv::Mat canny_chs[]  = {canny_imgf, canny_imgf, canny_imgf};

    cv::merge(canny_chs,3,canny_img3c);

    cv::Mat cartoonf;
    cartoon.convertTo(cartoonf,CV_32FC3);

    cv::multiply(cartoonf,canny_img3c,cartoonf);
    

    cartoonf.convertTo(cartoon,CV_8UC3);

    cv::imshow("CARTPOON",cartoon);


    



    cv::waitKey(0);
    



    return 0;
}