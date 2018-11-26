% training phase begins
%Select databse folder
f=uigetdir;
di=dir(fullfile(f,'*.pgm'));
% Data matrix is obtained
ma=[];
for i=1:length(di)
    fi = fullfile(f,di(i).name);
    image=imread(fi);
    ma(:,i)=image(:);
end
% mA is weighted average data matrix
mA=ma-(sum(ma,2)/size(ma,2));
Am=transpose(mA);
Z=Am/(sqrt(size(ma,2)));
[u E v]=svd(Z);
% Principal component matrix calculation
P=[];
P_reduced=[];

for i=1:30
   P(:,i)=v(:,i); 
end
P_reduced=transpose(P);
Y=P_reduced*mA;
% training phase ends
% testing phase starts
% Select a image for testing
[tinp,path]=uigetfile();
name=fullfile(path,tinp);
Ximg=imread(name);
subplot(1,2,1)
imshow(Ximg);
title('Test Image');

inp=[];
inp(:,1)=Ximg(:);
Ximgavg=inp-(sum(ma,2)/104);
Ynew=P_reduced*Ximgavg;
val=[];
% 2-norm is used for matching and image which gives minimum 2 norm is
% provided as output.
for i=1:size(Y,2)
   n=norm(Ynew-Y(:,i));
   val(:,i)=n;
end

[mi,in]=min(val);
match=fullfile(f,di(in).name);
subplot(1,2,2)
imshow(match);
title('Matched Image');
allmi=[];
g=uigetdir;
gi=dir(fullfile(g,'*.pgm'));
% testing phase ends
% code for accuracy
count=0;

for i=1:length(gi)
    str=gi(i).name;
    gix = fullfile(g,str);
    imag=imread(gix);
    inp=[];
    inp(:,1)=imag(:);
    imgavg=inp-(sum(ma,2)/size(ma,2));
    Ynew=P_reduced*imgavg;
    val=[];

    for i=1:size(Y,2)
        n=norm(Ynew-Y(:,i));
        val(:,i)=n;
    end

    [mi,in]=min(val);
    str2=di(in).name;
    fprintf('Test Image %s - Matched Image %s\n',str,str2);
    if str(1)==str2(1)
        count=count+1;
    end
    
end
fprintf('Accuracy is %.3f\n',(count/length(gi))*100);