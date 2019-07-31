function [opt] = getDefaultOpts()
	opt = {};
	opt.expt_name = 'photoreal';
	opt.which_algs_paths = {'select_50_person1_1', 'select_50_person1_2'};         % paths to images generated by algoritms, e.g. {'my_alg','baseline_alg'}
	opt.vigilance_path = 'vigilance';       % path to vigilance images
	opt.gt_path = 'select_50_person1_0';                     % path to gt images
	opt.Nimgs = 50;                        % number of images to test
	opt.Npairs = 50;                        % number of paired comparisons per HIT
	opt.Npractice = 10;                     % number of practice trials per HIT (number of non-practice trials is opt.Npairs-opt.Npractice)
	opt.Nhits_per_alg = 1;                 % number of HITs per algorithm
	opt.vigilance_freq = 0.1;               % percent of trials that are vigilance tests
	opt.use_vigilance = false;               % include vigilance trials (obviously fake images to check that Turkers are paying attention)	
	opt.ut_id = '57369ffc573b8f88627fd20e5460e86d';                    % set this using http://uniqueturker.myleott.com/
	opt.base_url = 'https://raw.githubusercontent.com/zachluo/AMT/master/';                 % url where images to test are accessible as "opt.base_url/n.png", for integers n
	opt.instructions_file = './instructions_basic.html';        % instructions appear at the beginning of the HIT
	opt.short_instructions_file = './short_instructions_basic.html';  % short instructions are shown at the top of every trial
	opt.consent_file = './consent_basic.html';             % informed consent text appears the beginning of the HIT
	opt.im_height = 128;                    % dimensions at which to display the stimuli
	opt.im_width = 128;                     %
	opt.paired = false;                      % if true, then fake/n.jpg will be pitted against real/n.jpg; if false, fake/n.jpg will be pitted against real/m.jpg, for random n and m
end