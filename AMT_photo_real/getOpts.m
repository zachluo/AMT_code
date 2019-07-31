function [opt] = getOpts(expt_name)
	
	switch expt_name
        
		case 'example_expt'
			opt = getDefaultOpts();
		
		otherwise
			error(sprintf('no opts defined for experiment %s',expt_name));
	end
	
	opt.expt_name = expt_name;
end