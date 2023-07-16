function [ threshold ] = spike_threshold_new2( edata,col )
%Spike Threshold function
% Function for finding the spike threshold using the method described in Azouz & Gray (2000) PNAS
% edata is the data variable where the 1st column is time and all subsequent columns are membrane potentials
% col is the column variable of edata, i.e. the membrane potential trace, used in this function 
% to find the corresponding spike thresholds, note; col must be >2.

%Parameters definitions
V = edata(:,col);
T = edata(:,1);
%dV_time = T(1:end-1,1);


% Finding the 1st Derivative
dV = diff(V);
dV = smooth(dV);
plot(dV); %for testing

%Finding the peak of the derivative
[p,ind] = findpeaks(dV,'MinPeakHeight',2.5,'MinPeakDistance', 35); %use values between 1.2 to 3.0 depending on the recording 

%Finding the spike threshold for all spikes in the trace
if length(p)>=1
      dV_th= p*0.033; %Finding the cutoff of the derivative (used Azouz & Gray (200) cutoff; 0.033)
      for i=1:length(ind) %Finding the corresponding voltage for the cutoff of the derivative
          voltage=V((ind(i)-30):ind(i)); %taking the voltage before the peak voltage of derivative
          time=T((ind(i)-30):ind(i));
          y=abs(dV((ind(i)-30):ind(i))-dV_th(i)); %finding the value corresponding to cutoff
          [~, idx] = min(y);
          %threshold(i,1)=voltage(idx); %use if only the voltage is needed
          threshold(i,1)=time(idx);
          threshold(i,2)=voltage(idx);
          
      end
else
    threshold=[0];
end

% Finding threshold for 1st and last spike only
threshold_f(1,:)=threshold(1,:); %for the RAMP current protocol only
threshold_f(2,:)=threshold(end,:);

end

