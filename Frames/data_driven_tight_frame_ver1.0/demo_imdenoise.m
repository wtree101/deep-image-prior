clear all;close all;clc
addpath('./tool/');

%% Parameter setting of image denoising
sigma     	= 20;                            			% noise level
patchSize 	= 16; 										% patch size
stepSize  	= 1;                       					% overlap step of data   
trainnum	= 40000;									% the number of samples selected for learning
lambda_1  	= 3.4 * sigma;            					% lambda for learning dictionary
lambda_2  	= 2.7 * sigma;            					% lambda for denoising by learned dictionary
opts.nIter	= 30;										% number loop for constructing data-driven filter bank
opts.A 		= (1/patchSize)*ones(patchSize^2,1);		% pre-input filters  (must be orthogonal)

%% Generate simulated noisy image
randn('seed',2013); rand('seed',2013)
fileName  	= 'image/barbara.png';
clear_img 	= double(imread(fileName)); 				% read image   
[h, w] 	  	= size(clear_img);  						% image size
noisy_img 	= round(clear_img + sigma*randn(h, w)); 	% add noise
noisy_img(noisy_img > 255) = 255; 
noisy_img(noisy_img < 0)   = 0; 						% put the image into range [0,255]
PSNRinput 	= Psnr(clear_img, noisy_img); 				% PSNR of noisy image

tic;


%% Checking the correctness of pre-defined filter subset A 
A = opts.A;
if ~isempty(A)
	r = size(A, 2);
	temp = wthresh(A'*A - eye(r),'h',1e-14);
	if sum(temp(:)) > 0 
		error('The input A does not meet the requirement!');
	end
end


%% Generate collection of image patches
Data  		= im2colstep(noisy_img, [patchSize, patchSize], [stepSize, stepSize]);
rperm 		= randperm(size(Data, 2));
patchData 	= Data(:, rperm(1:trainnum));



%% Learning filter bank from image patches
learnt_dict  = filter_learning(patchData, lambda_1, opts);

%% Denoising image by using the tight frame derived from learned filter banks
im_out 		 = frame_denoising(noisy_img, learnt_dict, lambda_2);
PSNRoutput 	 = Psnr(clear_img, round(im_out));

% Plot image
%% Show clear image
figure(1),imshow(uint8(clear_img));
title('Clear Image');
%% Show noisy image
figure(2);imshow(uint8(noisy_img));
title(['Sigma=',num2str(sigma),' PSNR=' num2str(PSNRinput) 'dB']);
%% Show denoised image
figure(3);imshow(uint8(im_out));
title(['Learned Denoise PSNR=' num2str(PSNRoutput) 'dB']);
fprintf('PSNR of denoised image is %f \n', PSNRoutput);

toc;
