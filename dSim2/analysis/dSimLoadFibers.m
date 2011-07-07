function [fibers] = dSimLoadFibers(fibersFile)

fid = fopen(fibersFile,'rt');
n = 0;
while(~feof(fid))
    lineBuffer = fgetl(fid);
    if(~isempty(lineBuffer))
        str = strrep(lineBuffer,'INF','Inf');
        try
            [x,y,z,r]=strread(str,'%f%f%f%f','delimiter',' ');
            n = n+1;
            fibers(n,:) = [x y z r];
        catch
        end
    end
end
fclose(fid);

end


