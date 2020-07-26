function [type, X, Y, Xt, Yt, num_class] = readData(filename)

% assume this is already in the path so readData doesn't need to be
% launched from the src directory
%addpath('../datasets')

setenv('LC_ALL','C')

switch filename
    
    case 'ijcnn1'
        [Y,X] = libsvmread('./ijcnn1');
        type = 'bc';
        num_class = 2;
        [Yt,Xt] = libsvmread('./ijcnn1.t');
        

    
    case 'covtype2'
        [Y,X] = libsvmread('./covtype2');
        type = 'bc';
        num_class = 2;
        n = size(X,1);
        nt = floor(n*0.2);
        idx = randsample(n,nt);
        Xt = X(idx,:);
        Yt = Y(idx,:);
        X(idx,:) = [];
        Y(idx,:) = [];
end

if strcmp(type, 'bc')
    if min(Y) == 0
        Y = 2*Y-1;
        Yt = 2*Yt - 1;
    end
end

if strcmp(type, 'mc')
    if min(Y) == 0
        Y = Y+1;
        Yt = Yt+1;
    end
end
