#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

///////////////////////////////////////////////////

using namespace cv;

//courtesy of stack overflow :)
namespace patch { //std not registering as namespace for to_string(int)
    template <typename T> std::string to_string(const T& n) {
        std::ostringstream stm;
        stm << n;
        return stm.str();
    }
}

/////////////Library//Functions/////////////////////

void display_image(const Mat&, std::string);

void save_image(const Mat&, std::string);

bool is_grayscale(Mat);

void color2grey(Mat*);

Mat get_gaussian_kernel(const int&, const double&);

Mat get_gaussian_deriv(const char&, const int&, const double&);

void gradient_mag(Mat*, const int&, const double&);

void gradient_dir(Mat*, const int&, const double&);

Mat get_laplacian(const int&, const double&);

void filter(Mat*, const Mat&);

void difference_of_gaussians(const int&, Mat*, const double&);

void downsample(Mat*, const int&, const double&);

void quantize(Mat*, const int&);

void add_noise(Mat*, const double&, const double&);

void hist_make(const Mat&, Mat*);

void enhance(Mat*, std::string);

Mat get_synthetic_image();

void canny_edges(Mat*, const int&, const int&, const int&);

void Hough_Circles(Mat&);

//////////////Homework//Functions/////////////////

void synthetic_image();

void LoG_DoG(Mat&);

void stent(Mat&);

void deriv_gaussian(Mat&, const int&, const double&);

//////////////////////////////////////////////

int main(int argc, char** argv) {
    Mat image;
    image = imread(argv[1], 1);
    if(!image.data) {
        std::cout<< "No image data \n";
        std::cin.get(); //wait for key press
        return -1;
    }

    std::cout << "Which part would you like to run (input int)?" << std::endl;
    std::cout << "1: Synthetic Image" << std::endl;
    std::cout << "2: LoG and DoG" << std::endl;
    std::cout << "3: Stent Operations" << std::endl;
    std::cout << "4: Derivative of Gaussian" << std::endl;
    std::cout << "5: Hough Circle Transform" << std::endl;

    int num = 0;
    std::cout << "Input:\t";
    std::cin >> num;
    std::cout << std::endl;
    if(num == 1) {
        synthetic_image();
    } else if(num == 2) {
        Mat synthetic = get_synthetic_image();
        LoG_DoG(synthetic);
    } else if(num == 3) {
        stent(image);
    } else if(num == 4) {
        int filter_size[3] = {3,8,12}; //3x3
        int sigma[3] = {1,3,5};
        for(int i = 0; i < 3; ++i) {
            deriv_gaussian(image, filter_size[i], sigma[i]);
        }
    } else if(num == 5) {
        Hough_Circles(image);
    } else if(num == 6) {       //TEST CODE

    }

    return 0;
}

//////////////////////////////////////////////

void display_image(const Mat& image, std::string imname) {
    namedWindow(imname, WINDOW_AUTOSIZE);
    imshow(imname, image);
    waitKey(0);
}

void save_image(const Mat& image, std::string imname) {
    imwrite(imname, image); //save grayscale to file
    std::cout << imname << std::endl;
}

bool is_grayscale(Mat image) {
    if(image.type() == CV_8UC1) { //CV_8UC1 is enumerated 8 bit single channel unsigned matrix
        return true;
    } else {
        return false;
    }
}

void color2grey(Mat* image) { //using luminosity method
    Mat grayscale = Mat(image->rows, image->cols, CV_8UC1); //Mat constructor
    for(int r = 0; r < image->rows; r++) {
        for(int c = 0; c < image->cols; c++) { //Each pixel is an array of 3. We weight each color based on human eye sensitivity.
            int tmp = (image->at<Vec3b>(r, c)[0] * .11) + (image->at<Vec3b>(r, c)[1] * .59)+ (image->at<Vec3b>(r, c)[2] * .33);
            grayscale.at<uchar>(r,c) = tmp;
        }
    }
    *image = grayscale;
}

Mat get_gaussian_kernel(const int& size, const double& sigma) { //assumes the sigma and size for x,y is the same
    const int midpoint = size / 2;
    const double spread = 1. / (sigma*sigma*2);
    const double constant = 1. / (sigma*std::sqrt(2.*M_PI));

    std::vector<float> gauss;
    gauss.reserve(size); //preallocates the memory
    for(int i = 0;  i < size;  ++i) {
        double x = i - midpoint;
        gauss.push_back(constant*std::exp(-x*x*spread));
    } 

    float sum=0;
    Mat kernel = Mat::zeros(size, size, CV_32FC1); //matrix of 32 bit floats
    for(int i = 0; i < size; ++i) {
        float temp = gauss.at(i);
        for (int j = 0;  j < size;  ++j) {
            kernel.at<float>(i,j) = gauss.at(j) * temp;
            sum += kernel.at<float>(i,j);
        }
    }

    return kernel / sum;
}

Mat get_gaussian_deriv(const char& dx_dy, const int& size, const double& sigma) {
    const int midpoint = size / 2;
    const double spread = -1. / (sigma*sigma*2);
    const double constant = -1. / (sigma*sigma*sigma*std::sqrt(2.*M_PI));

    std::vector<float> gauss_deriv;
    gauss_deriv.reserve(size); //preallocates the memory
    for(int i = 0; i < size; ++i) {
        double x = i - midpoint;
        gauss_deriv.push_back(constant*x*std::exp(-x*x*spread));
    }

    Mat xkernel = Mat::zeros(size, size, CV_32FC1); //SUM IS ALWAYS 0
    Mat ykernel = Mat::zeros(size, size, CV_32FC1);
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            xkernel.at<float>(i,j) = gauss_deriv.at(j);
            ykernel.at<float>(j,i) = gauss_deriv.at(j); //is transpose of xkernel
        }
    }

    switch(dx_dy) {
        case 'x': return xkernel;
        case 'y': return ykernel;
        default: return xkernel;
    }
}

void gradient_mag(Mat* image, const int& size, const double& sigma) {
    Mat dx, dy;
    dx = dy = image->clone();

    filter(&dx, get_gaussian_deriv('x', size, sigma)); //Compute X gradient
    filter(&dy, get_gaussian_deriv('y', size, sigma)); //Compute Y gradient

    bool gray = is_grayscale(*image);
    for(int i = 0; i < image->rows; ++i) { //Compute Gradient Magnitude 
        for(int j = 0; j < image->cols; ++j) {
            if(!gray) {
                for(int k = 0; k < 3; ++k) {
                    int x_sqr = dx.at<Vec3b>(i,j)[k]*dx.at<Vec3b>(i,j)[k];
                    int y_sqr = dy.at<Vec3b>(i,j)[k]*dy.at<Vec3b>(i,j)[k];
                    image->at<Vec3b>(i,j)[k] = std::sqrt(x_sqr + y_sqr);
                }
            } else {
                int x_sqr = dx.at<uchar>(i,j)*dx.at<uchar>(i,j);
                int y_sqr = dy.at<uchar>(i,j)*dy.at<uchar>(i,j);
                image->at<uchar>(i,j) = std::sqrt(x_sqr + y_sqr);
            }
        }
    }
}

void gradient_dir(Mat* image, const int& size, const double& sigma) {
    Mat dx, dy;
    dx = dy = image->clone();

    filter(&dx, get_gaussian_deriv('x', size, sigma)); //Compute X gradient
    filter(&dy, get_gaussian_deriv('y', size, sigma)); //Compute Y gradient

    bool gray = is_grayscale(*image);
    for(int i = 0; i < image->rows; ++i) { //Compute Gradient Magnitude 
        for(int j = 0; j < image->cols; ++j) {
            if(!gray) {
                for(int k = 0; k < 3; ++k) { //atan on set of [-pi/2, pi/2] -> image set of [0, 255]
                    image->at<Vec3b>(i,j)[k] = 255*(2/M_PI)*(M_PI/2 + std::atan2(dy.at<Vec3b>(i,j)[k], dx.at<Vec3b>(i,j)[k]));
                }
            } else {
                image->at<uchar>(i,j) = 255*(2/M_PI)*(M_PI/2 + std::atan2(dy.at<uchar>(i,j), dx.at<uchar>(i,j)));
            }
        }
    }
}

Mat get_laplacian(const int& size, const double& sigma) {
    const int midpoint = size / 2;
    const double spread = -1. / (sigma*sigma*2);
    const double constant = -1. / (sigma*sigma*sigma*sigma*M_PI);

    float sum = 0;
    Mat laplacian = Mat::zeros(size, size, CV_32FC1);
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            double x = i - midpoint;
            double y = j - midpoint;

            laplacian.at<float>(i,j) = constant*(1 + spread*(x*x + y*y))*std::exp(spread*(x*x + y*y));
        }
    }

    return laplacian;
}

void filter(Mat* image, const Mat& filter) {
    //filter2D(input, output, depth(neg = 0), kernel(Mat float), anchor((-1,-1) = center), delta, border)
    filter2D(*image, *image, -1, filter, Point(-1,-1), 0, BORDER_DEFAULT); //convolution
}

void difference_of_gaussians(const int& id, Mat* image, const double& sigma) {
    if(!is_grayscale(*image)) {
        color2grey(image);
    }

    Mat gauss1, gauss2;
    gauss1 = image->clone();
    gauss2 = image->clone();
    filter(&gauss1, get_gaussian_kernel(3, sigma + .01));
    display_image(gauss1, "Gauss1");
    save_image(gauss1, "Gauss1_" + patch::to_string(id) + ".png");

    filter(&gauss2, get_gaussian_kernel(7, sigma - .01));
    display_image(gauss2, "Gauss2");
    save_image(gauss2, "Gauss2_" + patch::to_string(id) + ".png");

    for(int i = 0; i < image->rows; ++i) {
        for(int j = 0; j < image->cols; ++j) {
            image->at<uchar>(i,j) = gauss1.at<uchar>(i,j) - gauss2.at<uchar>(i,j);
        }
    } 
}

void downsample(Mat* image, const int& size, const double& sigma) { //outputs a greyscaled and downsampled image
    filter(image, get_gaussian_kernel(size, sigma));

    bool gray = is_grayscale(*image);
    Mat downsampled(Size(image->rows/2, image->cols/2), (gray) ? CV_8UC1 : CV_32FC1); //subsampling 
    for(int r = 0; r < image->rows/2; r++) { //using an averaging filter
        for(int c = 0; c < image->cols/2; c++) {
            if(!gray) {
                for(int k = 0; k < 3; ++k) {
                    downsampled.at<Vec3b>(r,c)[k] = (image->at<Vec3b>(2*r,2*c)[k] + image->at<Vec3b>(2*r,2*c+1)[k] + image->at<Vec3b>(2*r+1,2*c)[k] + image->at<Vec3b>(2*r+1,2*c+1)[k])/4;
                }
            } else {
                downsampled.at<uchar>(r,c) = (image->at<uchar>(2*r,2*c) + image->at<uchar>(2*r,2*c+1) + image->at<uchar>(2*r+1,2*c) + image->at<uchar>(2*r+1,2*c+1))/4;
            }
        }
    }

    *image = downsampled;
}

void quantize(Mat* image, const int& nlevels) { //integer division is easier bit kmeans produces a better result
    if(!is_grayscale(*image)) { //if grayscale
        color2grey(image);
    }

    int n = 256/nlevels; //creates scale factor
    for(int r = 0; r < image->rows; ++r) {
        for(int c = 0; c < image->cols; ++c) { //formula is I = (I/nlevels)*nlevels
            image->at<uchar>(r,c) /= n; //integer division rounds to nearest integer
            image->at<uchar>(r,c) *= n;
        }
    }
}

void add_noise(Mat* image, const double& mu, const double& sigma) {
    const int size = 255;
    const double spread = 1. / (sigma*sigma*2);
    const double constant = 1. / (sigma*std::sqrt(2.*M_PI));

    float sum = 0;
    std::vector<float> gauss;
    gauss.reserve(size); //preallocates the memory
    for(int z = 0; z < size; ++z) {
        gauss.push_back(constant*std::exp(-spread*(z-mu)*(z-mu)));
    }

    bool is_gray = is_grayscale(*image);
    for(int i = 0; i < image->rows; ++i) {
        for(int j = 0; j < image->cols; ++j) {
            if(!is_gray) {
                for(int k = 0; k < 3; ++k) {
                    image->at<Vec3b>(i,j)[k] = image->at<Vec3b>(i,j)[k] + 255*gauss.at(std::rand() % size);
                }
            } else {
                image->at<uchar>(i,j) = image->at<uchar>(i,j) + 255*gauss.at(std::rand() % size);
            }
        }
    }
}

void hist_make(const Mat& image, Mat* hist) { //with special thanks to stack overflow!
    int bins = 256;
    float range[] = {0, 256};
    const float* hist_range = {range};

    calcHist(&image, 1, 0, Mat(), *hist, 1, &bins, &hist_range, true, false);

    int hist_w = 512; 
    int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/bins );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(*hist, *hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    /// Draw for each channel
    for( int i = 1; i < bins; i++ ) {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist->at<float>(i-1)) ) ,
                Point( bin_w*(i), hist_h - cvRound(hist->at<float>(i)) ),
                Scalar( 255, 0, 0), 2, 8, 0  
            );
    }

    *hist = histImage;
}

void enhance(Mat* image, std::string imname) {
    if(!is_grayscale(*image)) {
        color2grey(image);
    }

    //Mat OG_hist, NEW_hist;

    //hist_make(*image, &OG_hist);
    //display_image(OG_hist, "OG_hist.png");
    //save_image(OG_hist, imname.substr(0,imname.length()-4) + "_OG_hist.png");

    equalizeHist(*image, *image);
    //display_image(*image, "equalized_" + imname);
    //save_image(*image, "equalized_" + imname);

    //hist_make(*image, &NEW_hist);
    //display_image(NEW_hist, "NEW_hist.png");
    //save_image(NEW_hist, imname.substr(0,imname.length()-4) + "_NEW_hist.png");

}

Mat get_synthetic_image() {
    //width-height-type-color
    Mat synthetic(250, 250, CV_8UC3, Scalar(255,255,255)); //create white image

    for(int i = 50; i < 200; i++) {
        for(int j = 50; j < 200; j++) {
            if(((i > 86) && (i < 163)) && ((j > 86) && (j < 163))) { //color yellow
                synthetic.at<Vec3b>(i,j)[0] = 0; //B
                synthetic.at<Vec3b>(i,j)[1] = 255; //R
                synthetic.at<Vec3b>(i,j)[2] = 255; //G
            } else { //color blue
                synthetic.at<Vec3b>(i,j)[0] = 255; //B
                synthetic.at<Vec3b>(i,j)[1] = 0; //R
                synthetic.at<Vec3b>(i,j)[2] = 0; //G
            }
        }
    }

    return synthetic;
}

void synthetic_image() { 
    Mat synthetic = get_synthetic_image();
    display_image(synthetic, "test synthetic"); //test code
    save_image(synthetic, "test_synthetic.png"); 

    double mu = 255/2; //middle of [0,255] for all channels
    double sigma[4] = {std::sqrt(10), std::sqrt(20), std::sqrt(40), std::sqrt(100)};
    for(int i = 0; i < 4; ++i) {
        Mat synth = synthetic.clone();
        add_noise(&synth, mu, sigma[i]);
        display_image(synth, "synthetic_" + patch::to_string(sigma[i])); //test code
        save_image(synth, "synthetic_" + patch::to_string(sigma[i]) + ".png"); 
    }
}

void LoG_DoG(Mat& image) {
    if(!is_grayscale(image)) {
        color2grey(&image);
    }

    double sigma[2] = {.5, 2};
    int size[2] = {3, 11};
    int weight[2] = {10,100};
    for(int c = 0; c < 2; c++) {
        //LoG
        Mat LoG = image.clone();
        Mat DoG = image.clone();
        filter(&LoG, get_laplacian(size[c], sigma[c])); 
        display_image(weight[c]*LoG, "LoG_" + patch::to_string(c+1));
        save_image(weight[c]*LoG, "LoG_" + patch::to_string(c+1) + ".png");

        //DoG
        difference_of_gaussians(sigma[c], &DoG, sigma[c]);
        display_image(DoG, "DoG_" + patch::to_string(c+1));
        save_image(DoG, "DoG_" + patch::to_string(c+1) + ".png");

        //Zero crossing
        Mat zcross = Mat::zeros(DoG.rows, DoG.cols, CV_8UC1);   //ADD LOCAL CROSSINGS
        for(int i = 0; i < DoG.rows; i++) {
            for(int j = 0; j < DoG.cols; j++) {
                if(DoG.at<uchar>(i,j) > 225) {
                    zcross.at<uchar>(i,j) = 255;
                } else {
                    zcross.at<uchar>(i,j) = 0;
                }

                if((DoG.at<uchar>(i,j) - DoG.at<uchar>(i,j+1)) > 50 && 50 < (DoG.at<uchar>(i,j)- DoG.at<uchar>(i,j-1))) {
                    zcross.at<uchar>(i,j) = DoG.at<uchar>(i,j);
                }

                if((DoG.at<uchar>(i,j) - DoG.at<uchar>(i+1,j)) > 50 && 50 < (DoG.at<uchar>(i,j)- DoG.at<uchar>(i-1,j))) {
                    zcross.at<uchar>(i,j) = DoG.at<uchar>(i,j);
                }

                if((DoG.at<uchar>(i,j) - DoG.at<uchar>(i+1,j+1)) > 50 && 50 < (DoG.at<uchar>(i,j)- DoG.at<uchar>(i+1,j-1))) {
                    zcross.at<uchar>(i,j) = DoG.at<uchar>(i,j);
                }

                if((DoG.at<uchar>(i,j) - DoG.at<uchar>(i-1,j+1)) > 50 && 50 < (DoG.at<uchar>(i,j)- DoG.at<uchar>(i-1,j-1))) {
                    zcross.at<uchar>(i,j) = DoG.at<uchar>(i,j);
                }
            }
        }

        display_image(zcross, "Zero_Crossing_" + patch::to_string(c+1));
        save_image(zcross, "Zero_Crossing_"+ patch::to_string(c+1) + ".png");
    }
}

void stent(Mat& image) {
    if(!is_grayscale(image)) {
        color2grey(&image);
    }

    Mat enh = image.clone();
    Mat lapl = image.clone();
    Mat diff = image.clone();
    //enhance OG image
    enhance(&enh, "stent.png");
    display_image(enh, "stent_a");
    save_image(enh, "stent_a.png");

    //add LoG to enhanced image
    filter(&lapl, get_laplacian(7, .8));
    display_image(50*lapl, "TEST");
    diff = enh + 50*lapl;
    display_image(diff, "stent_b");
    save_image(diff, "stent_b.png");

    //blur w/ 3x3 blur
    filter(&image, get_gaussian_kernel(3, 7));
    display_image(image, "TEST");
    diff = diff - image;
    display_image(diff, "stent_c_blur");
    save_image(diff, "stent_c_blur.png");

    //add difference to OG image
    image = image + diff;
    display_image(image, "stent_c");
    save_image(image, "stent_c.png");
}

void deriv_gaussian(Mat& image, const int& size, const double& sigma) {
    Mat dx, dy;
    dx = dy = image.clone();

    filter(&dx, get_gaussian_deriv('x', size, sigma)); //Compute X gradient
    display_image(dx, "dx_Lenna_" + patch::to_string(sigma) + ".png");
    save_image(dx, "dx_Lenna_" + patch::to_string(sigma) + ".png");

    filter(&dy, get_gaussian_deriv('y', size, sigma)); //Compute Y gradient
    display_image(dy, "dy_Lenna_" + patch::to_string(sigma) + ".png");
    save_image(dy, "dy_Lenna_" + patch::to_string(sigma) + ".png");

    gradient_mag(&image, size, sigma);
    display_image(image, "grad_Lenna_" + patch::to_string(sigma) + ".png");
    save_image(image, "grad_Lenna_" + patch::to_string(sigma) + ".png");
}

void canny_edges(Mat* image, const int& size, const int& thresh1, const int& ratio) {
    Canny(*image, *image, thresh1, thresh1 * ratio, size); //intend to code this
}

void Hough_Circles(Mat& image) {
    Mat hough = image.clone(); //convert to grayscale
    if(!is_grayscale(hough)) {
        color2grey(&hough);
    }

    filter(&hough, get_gaussian_kernel(5, 7));

    canny_edges(&hough, 3, 225, 3); //gradient with canny edge detectos
    display_image(hough, "Canny");
    save_image(hough, "Canny.png");

    //implement circle hough transform and canny
    int minr = 0; //min/max radii for circles in an image
    int maxr = (hough.rows > hough.cols) ? (hough.cols / 2) : (hough.rows / 2);
    int maxdeg = 360;
    double deg_rad = M_PI / 180;
    int r = 90; //guess

    Mat accumulator = Mat::zeros(hough.rows, hough.cols, CV_8UC1);
    for(int i = 0; i < hough.rows; ++i) { //fill hough space
        for(int j = 0; j < hough.cols; ++j) {
            if(hough.at<uchar>(i,j) == 255) {
                //for(int r = 0; r < maxr - minr; ++r) {
                    for(int t = 0; t < maxdeg; ++t) {
                        int a = i - r*std::sin(t*deg_rad);
                        int b = j - r*std::cos(t*deg_rad);
                        if( accumulator.at<uchar>(a,b) != 255) {
                            accumulator.at<uchar>(a,b)++;
                        }
                    } display_image(10*accumulator, "x");
                //}
            }
        }
    }save_image(accumulator, "x.png");
/*
    GaussianBlur(hough, hough, Size(9,9), 2, 2);
    HoughCircles(hough, circles, HOUGH_GRADIENT, 1, hough.rows/8, 150, 50, 0, 0);
*/
    std::vector<Vec3f> circles;
    for( size_t i = 0; i < circles.size(); i++ ) { //draw circles on color image
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));// circle center
        int radius = cvRound(circles[i][2]);
        circle(image, center, 3, Scalar(0,255,0), -1, 8, 0);// circle outline
        circle(image, center, radius, Scalar(0,0,255), 3, 8, 0);
    }

    display_image(image, "Hough_Transform");
    save_image(image, "Hough_Transform.png");   
}
