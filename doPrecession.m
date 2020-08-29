function [flat, precessed] = doPrecession(atoms, n, alpha, system_conf, input_multislice)
    input_multislice.spec_atoms = double(atoms);
    output_multislice = il_MULTEM(system_conf, input_multislice);
    flat = output_multislice.data(1).m2psi_tot;
    
    precessed = zeros(n+1,size(flat,1), size(flat,2));    
    precessed(1,:,:) = flat;

    alpha = alpha*pi/180;
    Ra = [1,0,0;0,cos(alpha),-sin(alpha);0,sin(alpha),cos(alpha)];
    x = atoms(:,2:4); c = mean(x, 1);
    N = 2^n; theta = linspace(0,2*pi*N/(N+1),N) + 1; % random offset in radians
    
    for i=1:N
        t = theta(i);
        Rt = [cos(t),-sin(t),0;sin(t),cos(t),0;0,0,1];
        R = Rt'*Ra*Rt;
        atoms(:,2:4) = sortrows((x-c)*R'+c,3);

        input_multislice.spec_atoms = double(atoms);
        output_multislice = il_MULTEM(system_conf, input_multislice);
        dp = output_multislice.data(1).m2psi_tot;
        
        for j=1:n
            if mod(i-1,2^(n-j)) == 0
                precessed(j+1,:,:) = precessed(j+1,:,:) + reshape(dp,1,size(flat,1), size(flat,2));
            end
        end
    end
end