%this finds the height of the matrix
Biome=data;
dims=size(Biome);
height=dims(1,1);

score=zeros(1,5);
L=0;
Acc_KNN=zeros(1,10);
KNN=zeros(1,10);

while L < 10
    k=1;
    while k<=5 %this repeats for this many iterations
        K=1+L*2; %K value in K-nearest neighbor
        KNN(L+1)=K;
        [m,n] = size(Biome) ;
        P = 0.95 ; %Percent of rows being placed in training group
        idx = randperm(m)  ;
        train = Biome(idx(1:round(P*m)),:) ;
        test = Biome(idx(round(P*m)+1:end),:) ;
        sz_tr = size(train); h_tr=sz_tr(1);
        sz_tes = size(test);  h_tes=sz_tes(1);
        class=train(:,1);
        
        dist1=zeros(h_tes+1,h_tr);
        i=1;
        j=1;
        guess=zeros(h_tes,1);
        
        for i=1:h_tes
            for j=1:h_tr
                dist1(i+1,j)=norm(test(i,[2:end])-train(j,[2:end])); %Euclidian Distance
            end
            dist1(1,:)=class';
            M=[dist1(1,:);dist1(i+1,:)];
            M=M';
            M=sortrows(M,2);
            Mc=M(:,1);
            guess(i,1)=mode(Mc(1:K)); %choose K nearest neighbors as guess
        end
        
        acc=zeros(h_tes,1);
        for j=1:h_tes
            if guess(j)==test(j,1);
                acc(j)=1;
            else
                acc(j)=0;
            end
        end
        score(k)=sum(acc)/length(acc);
        k=k+1;
    end
    Acc_KNN(L+1)=mean(score);
    L=L+1;
end
figure
plot(KNN,Acc_KNN)
title('# of KNN vs Accuracy (Color)','FontSize', 20)
xlabel('# of Nearest Neighbors, KNN','FontSize', 14)
ylabel('Accuracy %','FontSize', 14)
text(.3*max(KNN),.99*max(Acc_KNN),['Test Group Size = 5% of Training Group Size'])