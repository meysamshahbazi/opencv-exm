#include "opencv2/opencv.hpp"
#include <iostream>
#include <math.h>



int main( int argc, const char **argv)
{
    cv::Mat src;
    
    
    src = cv::imread(argv[1]);
    std::istringstream ss(argv[2]);
    int nb_cluster;
    //  = static_cast<int>(argv[2]);
    ss>>nb_cluster;


    if(src.empty())
    {
        std::cerr<<"Image Dose Not exist in path: "<<argv[1]<<std::endl;
        return -1;
    }

    cv::namedWindow("SRC");
    cv::imshow("SRC",src);

    cv::Scalar color_tab[]=
    {
        cv::Scalar(0,0,255),
        cv::Scalar(0,255,0),
        cv::Scalar(255,0,0),
        cv::Scalar(0,255,255),
        cv::Scalar(255,0,255)
    };




    int width = src.cols;
    int height = src.rows;
    int dims = src.channels();

    int nb_samples = width*height;

    cv::Mat points(nb_samples,dims,CV_64FC1);

    cv::Mat labels;
    cv::Mat res = cv::Mat::zeros(src.size(),CV_8UC3);

    

    int index = 0;
    
    for( int row = 0; row<height;row++)
    {
        for (int col = 0; col<width; col++)
        {
            index = row*width + col;
            cv::Vec3b rgb = src.at<cv::Vec3b>(row,col);
            points.at<double>(index,0) = static_cast<int>(rgb[0]);
            points.at<double>(index,1) = static_cast<int>(rgb[1]);
            points.at<double>(index,2) = static_cast<int>(rgb[2]);
        }
    }

    cv::Ptr<cv::ml::EM> em_model = cv::ml::EM::create();

    em_model->setClustersNumber(nb_cluster);

    em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_SPHERICAL);

    em_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT,100,0.1));

    em_model->trainEM(points,cv::noArray(),labels,cv::noArray());

    cv::Mat sample(1,dims,CV_64FC1);

    int r = 0, g = 0, b = 0;


    for (int row  = 0;row<height;row++)
    {
        for( int col = 0; col< width; col++)
        {

            index = row*width+col;
            b = src.at<cv::Vec3b>(row,col)[0];
            g = src.at<cv::Vec3b>(row,col)[1];
            r = src.at<cv::Vec3b>(row,col)[2];

            sample.at<double>(0,0) = static_cast<double>(b);
            sample.at<double>(0,1) = static_cast<double>(g);
            sample.at<double>(0,2) = static_cast<double>(r);

            int response = cvRound(em_model->predict2(sample,cv::noArray())[1]);


            cv::Scalar c = color_tab[response];
            res.at<cv::Vec3b>(row, col)[0] = c[0];
            res.at<cv::Vec3b>(row, col)[1] = c[1];
            res.at<cv::Vec3b>(row, col)[2] = c[2];


        }
    }

    cv::imshow("EM-RES",res);

    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;



}


