%this finds the height of the matrix
Biome=data;
dims=size(Biome);
height=dims(1,1);


score=zeros(1,5);
L=1;
Acc_P_Value=zeros(1,10);
P_Size=zeros(1,10); 

while L <= 10
    k=1;
    while k<=5 %this repeats for this many iterations
        K=5; %K value in K-nearest neighbor
        [m,n] = size(Biome) ;
        P = 1-L*.01 ; %Percent of rows being placed in training group
        P_Size(L)=P;
        idx = randperm(m)  ;
        train = Biome(idx(1:round(P*m)),:);
        test = Biome(idx(round(P*m)+1:end),:);
        sz_tr = size(train); h_tr=sz_tr(1);
        sz_tes = size(test);  h_tes=sz_tes(1);
        class=train(:,1);
        
        dist2=zeros(h_tes+1,h_tr);
        i=1;
        j=1;
        guess=zeros(h_tes,1);
        
        for i=1:h_tes
            for j=1:h_tr
                dist2(i+1,j)=norm(test(i,[2:end])-train(j,[2:end])); %Euclidian Distance
            end
            dist2(1,:)=class';
            M=[dist2(1,:);dist2(i+1,:)];
            M=M';
            M=sortrows(M,2);
            class=M(:,1);
            guess(i,1)=mode(class(1:K)); %choose K nearest neighbors as guess
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
    Acc_P_Value(L)=mean(score);
    L=L+1;
end
figure
plot(P_Size,Acc_P_Value)
title('Valdiation Group Size vs Accuracy (Color)','FontSize', 20)
xlabel('Valdiation Group Size as % of Total','FontSize', 14)
ylabel('Accuracy %','FontSize', 14)
text(.6*max(P_Size),.99*max(Acc_P_Value),['KNN, N is held at 5'])