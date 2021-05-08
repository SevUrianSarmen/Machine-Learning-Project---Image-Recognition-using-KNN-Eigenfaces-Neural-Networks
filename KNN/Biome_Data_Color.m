clc; clear;

scale=150; %this value will scale all images in code

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Folder=[pwd,'\','seg_test','\','buildings']; %assign folder name for pool of images
S=dir(Folder); %list all file in folder to S
B = {S.name};
B(1:2)=[]; %remove first two lines of junk data

count=1;

for i=1:length(B)
    im_path=[Folder,'\',B(i)];
    im_path=cell2mat(im_path);
    I = imread(im_path);
    I_gray = I;
    I_gray = im2double(I_gray);
    I_flat = reshape(I_gray, 1, []);
%     I_flat=I_gray(:); %convert image matrix to column vector
%     I_flat=I_flat';
    %         I_flat=[0 0 I_flat];
    B_dat(i,:)=I_flat;
    %     T_dat(i,2)=count;
    count=count+1;
end
% B_dat( ~any(B_dat,2), : ) = [];
sz=size(B_dat);
class1=ones(sz(1),1).*1;    %identifies group as class 1
CL1=[class1,B_dat];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Folder=[pwd,'\','seg_test','\','forest']; %assign folder name for pool of images
S=dir(Folder); %list all file in folder to S
B = {S.name};
B(1:2)=[]; %remove first two lines of junk data

count=1;

for i=1:length(B)
    im_path=[Folder,'\',B(i)];
    im_path=cell2mat(im_path);
    I = imread(im_path);
    I_gray = I;
    I_gray = im2double(I_gray);
    I_flat = reshape(I_gray, 1, []);
%     I_flat=I_gray(:); %convert image matrix to column vector
%     I_flat=I_flat';
    %         I_flat=[0 0 I_flat];
    B_dat(i,:)=I_flat;
    %     T_dat(i,2)=count;
    count=count+1;
end
% B_dat( ~any(B_dat,2), : ) = [];
sz=size(B_dat);
class2=ones(sz(1),1).*2;    %identifies group as class 2
CL2=[class2,B_dat];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Folder=[pwd,'\','seg_test','\','glacier']; %assign folder name for pool of images
S=dir(Folder); %list all file in folder to S
B = {S.name};
B(1:2)=[]; %remove first two lines of junk data

count=1;

for i=1:length(B)
    im_path=[Folder,'\',B(i)];
    im_path=cell2mat(im_path);
    I = imread(im_path);
    I_gray = I;
    I_gray = im2double(I_gray);
    I_flat = reshape(I_gray, 1, []);
%     I_flat=I_gray(:); %convert image matrix to column vector
%     I_flat=I_flat';
    %         I_flat=[0 0 I_flat];
    B_dat(i,:)=I_flat;
    %     T_dat(i,2)=count;
    count=count+1;
end
% B_dat( ~any(B_dat,2), : ) = [];
sz=size(B_dat);
class3=ones(sz(1),1).*3;    %identifies group as class 3
CL3=[class3,B_dat];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Folder=[pwd,'\','seg_test','\','mountain']; %assign folder name for pool of images
S=dir(Folder); %list all file in folder to S
B = {S.name};
B(1:2)=[]; %remove first two lines of junk data

count=1;

for i=1:length(B)
    im_path=[Folder,'\',B(i)];
    im_path=cell2mat(im_path);
    I = imread(im_path);
    I_gray = I;
    I_gray = im2double(I_gray);
    I_flat = reshape(I_gray, 1, []);
%     I_flat=I_gray(:); %convert image matrix to column vector
%     I_flat=I_flat';
    %         I_flat=[0 0 I_flat];
    B_dat(i,:)=I_flat;
    %     T_dat(i,2)=count;
    count=count+1;
end
% B_dat( ~any(B_dat,2), : ) = [];
sz=size(B_dat);
class4=ones(sz(1),1).*4;    %identifies group as class 4
CL4=[class4,B_dat]; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Folder=[pwd,'\','seg_test','\','sea']; %assign folder name for pool of images
S=dir(Folder); %list all file in folder to S
B = {S.name};
B(1:2)=[]; %remove first two lines of junk data

count=1;

for i=1:length(B)
    im_path=[Folder,'\',B(i)];
    im_path=cell2mat(im_path);
    I = imread(im_path);
    I_gray = I;
    I_gray = im2double(I_gray);
    I_flat = reshape(I_gray, 1, []);
%     I_flat=I_gray(:); %convert image matrix to column vector
%     I_flat=I_flat';
    %         I_flat=[0 0 I_flat];
    B_dat(i,:)=I_flat;
    %     T_dat(i,2)=count;
    count=count+1;
end
% B_dat( ~any(B_dat,2), : ) = [];
sz=size(B_dat);
class5=ones(sz(1),1).*5;    %identifies group as class 5
CL5=[class5,B_dat];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Folder=[pwd,'\','seg_test','\','street']; %assign folder name for pool of images
S=dir(Folder); %list all file in folder to S
B = {S.name};
B(1:2)=[]; %remove first two lines of junk data

count=1;

for i=1:length(B)
    im_path=[Folder,'\',B(i)];
    im_path=cell2mat(im_path);
    I = imread(im_path);
    I_gray = I;
    I_gray = im2double(I_gray);
    I_flat = reshape(I_gray, 1, []);
%     I_flat=I_gray(:); %convert image matrix to column vector
%     I_flat=I_flat';
    %         I_flat=[0 0 I_flat];
    B_dat(i,:)=I_flat;
    %     T_dat(i,2)=count;
    count=count+1;
end
% B_dat( ~any(B_dat,2), : ) = [];
sz=size(B_dat);
class6=ones(sz(1),1).*6;    %identifies group as class 6
CL6=[class6,B_dat];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data=[CL1;CL2;CL3;CL4;CL5;CL6]; %create full data group
img=data;
img(:,1)=[];
