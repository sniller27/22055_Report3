function image = kernelProcessing(I, thresh, ks)
%best thresh is 90
%ks is kernelSize
Itemp = double(I);
Itemp2 = Itemp;
[rows, columns] = size(Itemp);
    for i = ks+1:rows-ks
        for j = ks+1:columns-ks
            Itemp2(i,j)= mean(mean(Itemp(i-ks:i+ks,j-ks:j+ks)>thresh))>0.5;
            
        end
    end


image = uint8(Itemp2.*255);
end