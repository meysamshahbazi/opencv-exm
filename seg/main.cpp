
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>


#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

const char* keys = 
{
    "{help h usage ? | | print this message}"
    "{@image || Image to process}"
    "{@lightPattern || Image light pattern to apply to image input}"
	"{lightMethod | 1 | Method to remove backgroun light, 0 differenec, 1 div, 2 no light removal' }"
	"{segMethod | 1 | Method to segment: 1 connected Components, 2 connectec components with stats, 3 find Contours }"
};

cv::Mat calculateLightPattern(cv::Mat img) 
{
    cv::Mat pat;
    cv::blur(img,pat,cv::Size(img.cols/3,img.rows/3));
    return pat;

}

static cv::Scalar randomColor(cv::RNG& rng)
{
	int icolor = (unsigned)rng;
	return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}




void ConnectedComponents(cv::Mat img)
{
    cv::Mat labels;
    int nb_obj = cv::connectedComponents(img, labels);

    cv::RNG rng(0xFFFFFFFF);
    cv::Mat out = cv::Mat::zeros(img.rows,img.cols, CV_8UC3);


    for (int i =1; i < nb_obj; i++)
    {
        cv::Mat mask = labels==i;
        out.setTo(randomColor(rng),mask);
    }

    cv::imshow("RES",out);


}

void ConnectedComponentsStats(cv::Mat img)
{
    cv::Mat labels, stats, centroid;
    int nb_obj = cv::connectedComponentsWithStats(img,labels,stats,centroid);
    
    cv::Mat out = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
    cv::RNG rng(0xFFFFFFFF);
    for (int i =1;i < nb_obj;i++)
    {
        if (stats.at<int>(i,cv::CC_STAT_AREA) > 20) 
        {
            cv::Mat mask = labels==i;
            out.setTo(randomColor(rng),mask);
        }
        
    }
    cv::imshow("RES-STAT",out);

}

void FindContoursBasic(cv::Mat img_thr,cv::Mat img )
{
    std::vector<std::vector<cv::Point> > countours;
    cv::findContours(img_thr,countours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    cv::Mat out = cv::Mat::zeros(img.rows, img.cols,CV_8UC3);
    cv::RNG rng(0xFFFFFFFF);

    for (int i = 0; i <countours.size(); i++)
    {
        cv::Scalar colr = randomColor(rng);
        cv::drawContours(img,countours,i,colr);
        cv::drawContours(out,countours,i,colr);
        
    }

    cv::imshow("CON",img);
    cv::imshow("out",out);


}

int main(int argc, const char ** argv)
{   
    cv::CommandLineParser parser(argc, argv, keys);

    parser.about("section 5. PhotoTool v1.0.0");
	//If requires help show
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}


    std::string img_file = parser.get<std::string>(0);
    std::string light_pattern_file = parser.get<std::string>(1);

    int method_light = parser.get<int>("lightMethod");
    int method_seg = parser.get<int>("segMethod");

    if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}

    cv::Mat img = cv::imread(img_file, 0);
    if (img.data == NULL)
    {
		std::cout << "Error loading image " << img_file << std::endl;
		return 0;
	}

    cv::Mat img_denois;
    cv::Mat img_smooth;

    cv::medianBlur(img,img_denois,3);
    cv::blur(img,img_smooth,cv::Size(3,3));

    cv::Mat light_pattern = cv::imread(light_pattern_file, 0);
	if (light_pattern.data == NULL){
		// Calculate light pattern
		light_pattern = calculateLightPattern(img_denois);
	}
    cv::medianBlur(light_pattern,light_pattern,3);
    // cv::namedWindow("img");
    cv::imshow("img",img);

    cv::imshow("LP",light_pattern);
    

    cv::Mat img_no_ligh;

    img_denois.copyTo(img_no_ligh);
    // cv::Mat aux;
    cv::Mat img_thr;
    if (method_light == 1)
    {
        cv::Mat img32,pat32;
        img.convertTo(img32, CV_32F);
        light_pattern.convertTo(light_pattern,CV_32F);

        img_no_ligh = 1 - (img32/light_pattern);

        img_no_ligh = img_no_ligh*255;
        img_no_ligh.convertTo(img_no_ligh,CV_8U);
        
        cv::threshold(img_no_ligh,img_thr,50,255,cv::THRESH_BINARY);

    }
    else if (method_light == 0)
    {
        img_no_ligh = light_pattern-img;
        cv::threshold(img_no_ligh,img_thr,50,255,cv::THRESH_BINARY);
    }
    else if (method_light == 2)
    {
        cv::threshold(img_no_ligh,img_thr,140,255,cv::THRESH_BINARY_INV);

    }
    cv::imshow("img_no_ligh",img_no_ligh);
    cv::Mat img_tmp = cv::imread(img_file);
    if (method_seg==1)
    {
        ConnectedComponents(img_thr);
        ConnectedComponentsStats(img_thr);
        FindContoursBasic(img_thr,img_tmp);
        
    }
    





    cv::waitKey(0);





    return 0;
}