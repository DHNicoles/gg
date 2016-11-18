#include "net_manager.h"

namespace slh
{

	/* Load the mean file in binaryproto format. */
	void NetManager::SetMean(const string& mean_file) {

		bool is_binaryproto = true;
		if (mean_file.find(".binaryproto") == string::npos)
			is_binaryproto = false;

		mean_ = cv::Mat(input_geometry_, CV_32FC3);
		int h = input_geometry_.height;
		int w = input_geometry_.width;
		//float* mean_data = (float *)mean_.data;
		std::vector<cv::Mat> channels;
		if (is_binaryproto)
		{
			BlobProto blob_proto;
			ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

			/* Convert from BlobProto to Blob<float> */
			Blob<float> mean_blob;
			mean_blob.FromProto(blob_proto);

			CHECK_EQ(mean_blob.channels(), num_channels_)
				<< "Number of channels of mean file doesn't match input layer.";
			CHECK_EQ(mean_blob.height(), h)
				<< "Height of channels of mean file doesn't match input layer.";
			CHECK_EQ(mean_blob.width(), w)
				<< "Width of channels of mean file doesn't match input layer.";

			/* The format of the mean file is planar 32-bit float BGR or grayscale. */
			float* data = mean_blob.mutable_cpu_data();

			/*
			for ( int i = 0; i < num_channels_; ++i)
			{
			for ( int r = 0; r < h; ++r )
			{
			for ( int c = 0; c < w; ++c )
			{
			mean_data[num_channels_*(r*w+c)+i] = data[r*w+c];
			}
			}
			data += h * w;
			}
			*/

			// for somekind of reason cv::merge always use gpu0...
			for (int i = 0; i < num_channels_; ++i) {
				// Extract an individual channel. 
				cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
				channels.push_back(channel);
				data += mean_blob.height() * mean_blob.width();
			}

			cv::merge(channels, mean_);
		}
		else
		{
			std::ifstream f_channel_means(mean_file.c_str());

			float channel_mean_val;
			/*
			while ( f_channel_means >> channel_mean_val )
			{
			std::fill( mean_data, mean_data + h * w, channel_mean_val );
			mean_data += h * w;
			}*/

			// will have wired gpu0 problem
			std::vector<cv::Mat> channels;
			while (f_channel_means >> channel_mean_val)
			{
				cv::Mat channel(input_geometry_, CV_32FC1, channel_mean_val);
				channels.push_back(channel);
			}
			cv::merge(channels, mean_); // wierd that cannot put this & channels out side if-else
		}
	}
	void NetManager::OnInit(const string& model_file, const string& trained_file, const string& mean_file, const int device_id)
	{

		google::SetStderrLogging(google::GLOG_FATAL);
		Caffe::set_mode(Caffe::CPU);
		device_id_ = device_id;
		// Load the network.
		net_.reset(new Net<float>(model_file, TEST));
		net_->CopyTrainedLayersFrom(trained_file);

		CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
		CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

		Blob<float>* input_layer = net_->input_blobs()[0];

		num_channels_ = input_layer->channels();
		CHECK(num_channels_ == 3 || num_channels_ == 1)
			<< "Input layer should have 1 or 3 channels.";
		input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

		/* Load the binaryproto mean file. */
		SetMean(mean_file);
		if (device_id_ >= 0)
			Caffe::SetDevice(device_id_);

		//Blob<float>* input_layer = net_->input_blobs()[0];

		input_layer->Reshape(1, num_channels_,
			input_geometry_.height, input_geometry_.width);
		/* Forward dimension change to all layers. */
		net_->Reshape();

		WrapInputLayer(&input_channels);
	}
	void NetManager::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
		Blob<float>* input_layer = net_->input_blobs()[0];
		int width = input_layer->width();
		int height = input_layer->height();
		float* input_data = input_layer->mutable_cpu_data();
		for (int i = 0; i < input_layer->channels(); ++i) {
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		}
	}

	void NetManager::Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels) {
		/* Convert the input image to the input image format of the network. */

		cv::Mat sample;
		if (img.channels() == 3 && num_channels_ == 1)
			cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
		else if (img.channels() == 4 && num_channels_ == 1)
			cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
		else if (img.channels() == 4 && num_channels_ == 3)
			cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
		else if (img.channels() == 1 && num_channels_ == 3)
			cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
		else
			sample = img;

		cv::Mat sample_resized;
		if (sample.size() != input_geometry_)
			cv::resize(sample, sample_resized, input_geometry_);
		else
			sample_resized = sample;

#ifdef ZERO_MASK
		cv::Rect logo_rect(0, 0, round(input_geometry_.width * W_MASK_RATIO), round(input_geometry_.height * H_MASK_RATIO));
		sample_resized(logo_rect) = cv::Scalar(0, 0, 0);
#endif

		cv::Mat sample_float;
		if (num_channels_ == 3)
			sample_resized.convertTo(sample_float, CV_32FC3);
		else
			sample_resized.convertTo(sample_float, CV_32FC1);

		cv::Mat sample_normalized;
		cv::subtract(sample_float, mean_, sample_normalized);


		cv::split(sample_normalized, *input_channels);

		CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
			== net_->input_blobs()[0]->cpu_data())
			<< "Input channels are not wrapping the input layer of the network.";
	}
	void NetManager::ForwardAndGetfeature(cv::Mat &img, vector<float>& featrueVec, const char* feat_layer_name)
	{
		for (size_t i = 0; i < img.cols * 0.4; i++){
			for (size_t j = 0; j < img.rows * 0.25; j++){
				if (img.channels() == 1){
					img.at<uchar>(j, i) = 255;
				}
				else if (img.channels() == 3){
					img.at<cv::Vec3b>(j, i)[0] = 255;
					img.at<cv::Vec3b>(j, i)[1] = 255;
					img.at<cv::Vec3b>(j, i)[2] = 255;
				}
			}
		}

		Preprocess(img, &input_channels);

		net_->Forward();
		//mark add here.was going to compute feature.
		const boost::shared_ptr<Blob<float> > feature_blob = net_->blob_by_name(feat_layer_name);
		int batch_size = feature_blob->num();
		int dim_features = feature_blob->count() / batch_size;
		const float* feature_blob_data;
		float total_feature = 0;
		bool normalize_flag = true;
		for (int n = 0; n < batch_size; ++n) {
			feature_blob_data = feature_blob->cpu_data() +
				feature_blob->offset(n);
			if (dim_features != featrueVec.size())
			{
				LOG(ERROR) << "dim_features are not equal to feature_dimensions";
				return;
			}
			for (int d = 0; d < dim_features; ++d) {
				total_feature += feature_blob_data[d] * feature_blob_data[d];
			}
			total_feature = sqrt(total_feature);

			if (normalize_flag == true){
				for (int d = 0; d < dim_features; ++d) {
					featrueVec[d] = feature_blob_data[d] / total_feature;
				}
			}
			else {
				for (int d = 0; d < dim_features; ++d) {
					featrueVec[d] = feature_blob_data[d];
				}
			}
		}//end for 
	}

}
