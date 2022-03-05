#include <iostream>
#include <string>
#include <sstream>
#include <cmath>

// OpenCV includes
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"

cv::Mat light_pat;
cv::Ptr<cv::ml::SVM> svm;

const char* keys =
{
	"{help h usage ? | | print this message}"
	"{@image || Image to classify}"
};


static cv::Scalar randomColor(cv::RNG& rng)
{
	int icolor = (unsigned)rng;
	return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

cv::Mat removeLight(cv::Mat img, cv::Mat pattern)
{
	cv::Mat aux;

	// Require change our image to 32 float for division
	cv::Mat img32, pattern32;
	img.convertTo(img32, CV_32F);
	pattern.convertTo(pattern32, CV_32F);
	// Divide the imabe by the pattern
	aux = 1 - img32 / pattern32;
	// Scale it to convert o 8bit format
	aux = aux * 255;
	// Convert 8 bits format
	aux.convertTo(aux, CV_8U);

	//equalizeHist( aux, aux );
	return aux;
}





cv::Mat preprocessImg(cv::Mat inp)
{
    cv::Mat res;
    cv::Mat img_denois, img_box_smooth;
    cv::medianBlur(inp,img_denois, 3);
    
    cv::Mat img_no_ligh;
    img_denois.copyTo(img_no_ligh);
    cv::blur(img_denois,light_pat,cv::Size(img_denois.cols/3,img_denois.rows/3));
    img_no_ligh = removeLight(img_denois,light_pat);

    


    cv::threshold(img_no_ligh,res, 30, 255, cv::THRESH_BINARY);
    
    return res;
}

std::vector< std::vector<float> >  ExtractFeatures(cv::Mat img,
                                                    std::vector<int> * left = NULL,
                                                    std::vector<int> * top = NULL)
{
    std::vector< std::vector<float> > out;
    std::vector< std::vector<cv::Point> > countours;
    std::vector<cv::Vec4i>  hierarchy;

    cv::Mat inp = img.clone();

    cv::findContours(inp, countours,hierarchy,cv::RETR_CCOMP,cv::CHAIN_APPROX_SIMPLE);
    if (countours.size() == 0)
    {
        return out;
    }

    cv::RNG rng(0);
    for (int i = 0; i <countours.size(); i++)
    {
        cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
        cv::drawContours(mask,countours,i,cv::Scalar(1),cv::FILLED,cv::LINE_8,hierarchy,1);
        cv::Scalar area_s = sum(mask);
        float area = area_s[0];

        if (area > 500)
        {
            cv::RotatedRect r = cv::minAreaRect(countours[i]);
            float width = r.size.width;
            float height = r.size.height;
            float ar = (width<height) ? height/width : width/height;

            std::vector<float> row;
            row.push_back(area);
            row.push_back(ar);
            out.push_back(row);
            if(left != NULL)
            {
                left->push_back((int) r.center.x);
            }
            if (top != NULL)
            {
                top->push_back((int) r.center.y);
            }
            
            cv::imshow("MASK",mask*255);
            cv::waitKey(200);
        }
    }
    return out;

}

bool readFolderAndExtractFeatures(std::string folder,int label, int nb_for_test, 
    std::vector<float> &data_train, std::vector<int> &lables_train,
    std::vector<float> &data_test,std::vector<float> &labels_test)
{
    cv::VideoCapture imgs;
    if (!imgs.open(folder))
    {
        std::cout<<"Can not Open the Folder images"<<std::endl;
        return false;

    }
    cv::Mat frame;
    int img_index = 0;
    while(imgs.read(frame))
    {
        cv::cvtColor(frame,frame,cv::COLOR_BGR2GRAY);
        frame = preprocessImg(frame);
        std::vector<std::vector<float>> features = ExtractFeatures(frame);
        for(int i =0; i <features.size();i++)
        {
            if (img_index>=nb_for_test)
            {
                data_train.push_back(features[i][0]);
                data_train.push_back(features[i][1]);
                lables_train.push_back(label);
            }
            else 
            {
                data_test.push_back(features[i][0]);
                data_test.push_back(features[i][1]);
                labels_test.push_back((float)label);
                
            }
        }
        img_index++;
    }
    return true;
}



int main (int argc, const char ** argv)
{
    std::vector<float> data_train;
    std::vector<int> lables_train;
    std::vector<float> data_test;
    std::vector<float> labels_test;

    int nb_for_test = 20;

    // cv::Mat frame = cv::imread(argv[1]);
    // cv::cvtColor(frame,frame,cv::COLOR_BGR2GRAY);
    
    // frame = preprocessImg(frame);
    // cv::imshow("frame",frame);
    // cv::waitKey(0);
    
    std::vector<std::vector<float>> features = ExtractFeatures(frame);
    // readFolderAndExtractFeatures()




    return 0;
}







