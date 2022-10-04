%% ELE6701A Project
% *Claudio Alkazzi: 2096201*
%% *Introduction*
% The goal of this project is to simulate the transmission of data through
% a channel and to experiment with different types of receivers. To achieve
% this, we pass our randomly generated input signals through the required
% channel, and we build three receivers: Maximum Likelihood (ML), Linear Minimum
% Mean Squared Error (LMMSE), and a Least Mean-Square adaptive filter
% (LMS).
%% *Part I - ML Detector*
%% a) Structure of ML Detector
% Here, we create a matrix holding all the possible 10-symbol sequences
% that can be sent, and we create the required channel and apply it to the
% previously created s matrix. The ML detector will then compare the
% received sequence with all possible sequences.

%Create s
N = 10;  %Number of symbols
M = 2^N;   %Number of possibilities (hypotheses)
s = zeros(M,N);
for j=1:N
   s_ = [-1*ones(M/2^j,1); ones(M/2^j,1)];  %temporary values of s
   s(:,j) = repmat(s_,2^(j-1),1);
end
clear s_

%Introduce Channel
sigma2 = 0.2;
H = diag(ones(N,1)) + diag(0.5*ones(N-1,1),-1);
z = H * s';     % Calculate effect of the channel
z = z'; % Transpose the result for comparison. 

%% b) Error Bounds
% We find an upper bound for the union error and the minimum distance
% error.

%Find error bounds
Pe_union = 0; 
max_error = 0; % Used for Pe_mindist
for i = 1:M
    for j = 1:M
        if i ~= j
            error = qfunc( sum((sqrt((s(j,:) - s(i,:)).^2)) / (2*sqrt(sigma2))) ); % Do not divide by 2
            Pe_union = Pe_union + error;

            if error > max_error
                max_error = error;
            end
        end
    end
end
Pe_union =  (1/M) * Pe_union;  % equiprobable -> pi_i are equal to 1/M
Pe_mindist = (M-1) * max_error;
fprintf('Union probability of error = %f\n', Pe_union);
fprintf('Minimum distance probability of error = %f\n', Pe_mindist);

% The union error bound gives a more accurate representation of the
% probability of error than the minimum distance. Since the minimum
% distance error bound is greater than 1, it is telling us that the
% probability of error is surely less than 1, which is a trivial response.

%% c) Testing the system
% We test the system and find the actual simulated probability of error. 
% Moreover, for efficiency, we randomly choose an element of z instead of s
% since we have already calculated them.

noTransmissions = 100000;
noDetectionErrors = 0;
noSymbolErrors = 0;
for count=1:noTransmissions
    k = randi(M); % Choose a random index to transmit
    y = z(k,:)+ sqrt(sigma2)*randn(1,N); % Take z not s for efficiency
    dist = zeros(M,1);
    for i=1:M
        dist(i) = sum((y-z(i,:)).^2, 2);
    end
    [v,j] = min(dist);
    
    %Check if error and number of wrong symbols
    isError = false;
    for i = 1:N
        if z(j,i) ~= z(k,i)
            noSymbolErrors = noSymbolErrors + 1;
            isError = true; % Mark that error exists
        end
    end
    if isError == true
        noDetectionErrors = noDetectionErrors + 1;
    end
    
end
PeDetecion_sim = noDetectionErrors / noTransmissions;
PeSymbol_sim = noSymbolErrors / (noTransmissions * N);
fprintf('Simulated probability of error = %f\n', PeDetecion_sim);
%% d) Probability of Error Per Symbol
% Using the previously written code, we can easily find the probability of
% error per symbol.
fprintf('Simulated probability of error per symbol = %f\n', PeSymbol_sim);

%% e) Complexity of ML Detector
% Although the Maximum Likelyhood detector is an optimal detector, it has
% to compare the sent signal to every possible signal. This makes this
% detector incredibly complex to implement, and this is evident in the
% long execution time of the code.

%% *Part II - LMMSE equaliser*
% Even though ML is an optimal detector, the high complexity provides
% incentive to look for less optimal but faster detectors. Therefore, we
% shift our attention to a linear alternative: the LMMSE detector.
%% a) Optimal Coefficients
% Using the MMSE criteria, we find the three sets of optimal coefficients
% for the different given values of delta. After that, we calculate the
% theoretical MSE for each set. 
% For our calculations, we note that E[x]=0, E[x^2]=1, E[n]=0, E[n^2]=0.2I.
% This gives E[y]=0 and E[y^2]=Ry to be calculated below.

N = 3;
H = [1  0.5 0   0
     0  1   0.5 0
     0  0   1   0.5];
e = [1 0 0 0; 0 1 0 0; 0 0 1 0];
mmse = zeros(1,N);
w = zeros(N);
for i=1:3
    Ry = H*H' + sigma2*diag(ones(N,1));
    w(i,:) = e(i,:)*H' * inv(Ry);
    mmse(i) = 1 - w(i,:)*Ry*w(i,:)';
    disp(strcat('delta=',num2str(i-1),': [a0 a1 a2] = ',num2str(w(i,:)),'     mmse=',num2str(mmse(i))));
end

% Take delta=1 from now on
w = w(2,:); % Choose w for delta=1
mmse = mmse(2); % Choose mmse for delta=1

%% b) Testing the System
% We take delta=1. We then simulate the system and calculate the
% experimental MSE and the probability of error before and after the
% decision module.

%Initialize buffer
x = zeros(N+1,1);
noTransmissions = 1000000;
noError = 0;
noError_y = 0;
mse = 0;
mse_y = 0;
for count = 1:noTransmissions
    %Create random symbol and add it to buffer
    for i = N+1:-1:2
        x(i) = x(i-1);
    end
    x(1) = 2*randi(2) - 3; % symbol = {+1, -1}
    
    y = H*x + sqrt(sigma2)*randn(N,1);
    mse_y = mse_y + (y(1) - x(1))^2;
    noError_y = noError_y + (sign(y(1)) ~= x(1));
    
    x_est = w*y;
    mse = mse + (x_est-x(2))^2;
    
    s_est = sign(x_est);
    noError = noError + (s_est ~= x(2));
end
Pe = noError / noTransmissions;
mse = mse / noTransmissions;
mse_y = mse_y / noTransmissions;
Pe_y = noError_y / noTransmissions;

% II-b).i
fprintf('Simulated MSE = %f\n', mse);
fprintf('Theoretical MSE = %f\n', mmse);
% The experimental MSE is very close to the theoretical one found above.

% II-b).ii
fprintf('Simulated probability of error per symbol for LMMSE = %f\n', Pe);
fprintf('Simulated probability of error per symbol for ML = %f\n', PeSymbol_sim);
% As a reminder, the simulated probability of error for ML is shown here.
% As we can see, LMMSE is more prone to error than ML. This is logical
% since ML is an optimal detector, whereas LMMSE is the optimal *linear*
% detector.

% II-b).iii
fprintf('MSE before decision module = %f\n', mse_y);
fprintf('Probability of error per symbol before equalizer = %f\n', Pe_y);
% We can see that the MSE and probability of error per symbol at the output
% of the channel are both higher than the ones we found before. This shows
% the effectiveness of our equalizer and decision module.
%% *Part III - LMS Detector*
% In this part, instead of predetermining the values of the coefficients,
% we create an adaptive filter which finds them automatically. To achieve
% this, we create a LMS detector. First, we use different values of mu to
% train and plot the learning curve, then we use mu=0.01 to perform
% predictions and detect the incoming signal.
%% a) Training Curves
% We take mu = 0.01,0.05,0.1 and 500 symbols to train the model for
% delta=1. We take an initial guess of w=[5 5 5] to make the learning
% process visible in the plots.
N=3;
mu = [0.01, 0.05, 0.1];

% Train filter
noTransmissions = 500;
L = 1000; % Number of experiments: Used to average the error
err = zeros(1,noTransmissions);
%hold on
for m = 1:length(mu) 
    w = 5*ones(1,N);
    x = zeros(N+1,1);
    e = zeros(L,noTransmissions);
    for j = 1:L
        for count = 3:noTransmissions
            % Create random symbol and add it to buffer
            for i = N+1:-1:2
                x(i) = x(i-1);
            end
            x(1) = 2*randi(2) - 3; % symbol = {+1, -1}
            y = H*x + sqrt(sigma2)*randn(N,1);

            e(j,count) = x(2) - w*y;
            w = w + mu(m)*y'*e(j,count);
        end
    end
    err = 1/L * sum(e.^2, 1);
    figure,
    plot(err(2:end)); title(strcat('Training for mu=',num2str(mu(m))));
    ylabel('MSE');
end
%hold off
%% b) Training and Detection
% We retrain the model with 50 then 500 symbols for mu=0.01, and we compare
% the results.

%Training for 50 symbols
noTransmissions = 50;
w = 5*ones(1,N);
x = zeros(N+1,1);
e = zeros(1,noTransmissions);
for count = 3:noTransmissions
    % Create random symbol and add it to buffer
    for i = N+1:-1:2
        x(i) = x(i-1);
    end
    x(1) = 2*randi(2) - 3; % symbol = {+1, -1}
    y = H*x + sqrt(sigma2)*randn(N,1);

    e(count) = x(2) - w*y;
    w = w + mu(m)*y'*e(count);
end
%Perform predictions
noTransmissions = 1000000;
noErrors_lms = 0;
e = zeros(1,noTransmissions);
for count = 1:noTransmissions
    % Create random symbol and add it to buffer
    for i = N+1:-1:2
        x(i) = x(i-1);
    end
    x(1) = 2*randi(2) - 3; % symbol = {+1, -1}
    y = H*x + sqrt(sigma2)*randn(N,1);
    
    x_est = w*y;
    e(count) = x(2) - x_est;
    noErrors_lms = noErrors_lms + (sign(x_est) ~= x(2));
end
mse_lms = sum(e.^2)/noTransmissions;
Pe_lms = noErrors_lms/noTransmissions;
fprintf('MSE for LMS after training with 50 symbols (mu=0.01) = %f\n', mse_lms);
fprintf('Probability of error per symbol for LMS after training with 50 symbols (mu=0.01) = %f\n', Pe_lms);

%Training for 500 symbols
noTransmissions = 500;
w = 5*ones(1,N);
x = zeros(N+1,1);
e = zeros(1,noTransmissions);
for count = 3:noTransmissions
    % Create random symbol and add it to buffer
    for i = N+1:-1:2
        x(i) = x(i-1);
    end
    x(1) = 2*randi(2) - 3; % symbol = {+1, -1}
    y = H*x + sqrt(sigma2)*randn(N,1);

    e(count) = x(2) - w*y;
    w = w + mu(m)*y'*e(count);
    if count>400 && e(count)<0.24
        break
    end
end
%Perform predictions
noTransmissions = 1000000;
noErrors_lms = 0;
e = zeros(1,noTransmissions);
for count = 1:noTransmissions
    % Create random symbol and add it to buffer
    for i = N+1:-1:2
        x(i) = x(i-1);
    end
    x(1) = 2*randi(2) - 3; % symbol = {+1, -1}
    y = H*x + sqrt(sigma2)*randn(N,1);
    
    x_est = w*y;
    e(count) = x(2) - x_est;
    noErrors_lms = noErrors_lms + (sign(x_est) ~= x(2));
end
mse_lms = sum(e.^2)/noTransmissions;
Pe_lms = noErrors_lms/noTransmissions;
fprintf('MSE for LMS after training with 500 symbols (mu=0.01) = %f\n', mse_lms);
fprintf('Probability of error per symbol for LMS after training with 500 symbols (mu=0.01) = %f\n', Pe_lms);

%Conclusion III-b) goes here

%% *Conclusion*
% We generate random signals and send them through a channel, and we use
% ML, LMMSE, and LMS detectors to properly detect these transmitted
% signals. We see that ML performs best, having the lowest probability of
% error, but this comes at the cost of a complex algorithm. We then use
% LMMSE and see that it in fact performs well, even if it doesn't provide
% the same results as ML. Finally, we the benefits of the LMS adaptive
% filter when we are not able to calculate the necessary coefficients to
% perform predictions, and we see the effects of having different values of
% mu on the training of the filter.