function y=thomas(aa,bb,cc,f)
    n = length(f);
    v = zeros(n,1);   
    y = v;
    
    w = aa(1);
    y(1) = f(1)/w;
for j=2:n
    v(j-1) = bb(j)/w;
    w = aa(j) - cc(j)*v(j-1);
    y(j) = ( f(j) - cc(j)*y(j-1) )/w;
end    
    

for j=n-1:-1:1
    y(j) = y(j) - v(j)*y(j+1);
end

end
