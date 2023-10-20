import os
import subprocess
import time


def get_feature(poly_file, config):
    with open(poly_file) as f:
        lines = f.readlines()
        for index in range(len(lines)):
            lines[index] = lines[index].strip()
        polys = ''.join(lines)
    # some polys: p1, p2, while some maybe are [p1, p2] or {p1, p2}
    if polys[0] == '[' or polys[0] == '{':
        polys = polys[1:-1]

    time_dir = 'datasets/' + config.dataset + '/projection/time/'  + config.file + '.time'
    feature_dir = 'datasets/' + config.dataset + '/projection/txt/' + config.file + '.f'
    maple_dir = 'datasets/' + config.dataset + '/projection/mpl/' + config.file + '.mpl'

    if not os.path.exists(feature_dir):
        with open(maple_dir, 'w') as f:
            f.write('with(ProjectionCAD_utils):\n')
            f.write('polys:=[' + polys + ']:\n')
            f.write('vars:=indets(polys):\n')
            f.write("kernelopts(printbytes=false):\n")
            f.write('features:=get_features(polys, convert(vars, list)):\n')
            f.write(f'writestat(\"{feature_dir}\", [features, vars]):\n')      
            f.write('quit;\n')

        cmd = 'maple ' + maple_dir
        try:
            ori_time = time.time()
            p = subprocess.Popen(cmd, shell=True)
            p.wait(timeout=config.remain_time)
            ori_time = time.time() - ori_time
            with open(time_dir, 'w') as f:
                f.write(str(ori_time))
        
            if os.path.exists(feature_dir):
                with open(feature_dir) as f:
                    line = f.readline().strip()
                if line[:2] == '[f':
                    with open(feature_dir, 'w') as f:
                        f.write('error')
            else:
                with open(feature_dir, 'w') as f:
                    f.write('error')

            # kill_mserver = 'ps -ef | grep ' + maple_dir + ' | grep -v grep | awk \'{print $2}\' | xargs kill -s 9'
            # p1 = subprocess.Popen(kill_mserver, shell=True)
            # p1.wait()


        except subprocess.TimeoutExpired:
            kill_mserver = 'ps -ef | grep ' + maple_dir + ' | grep -v grep | awk \'{print $2}\' | xargs kill -s 9'
            p1 = subprocess.Popen(kill_mserver, shell=True)
            p1.wait()
            with open(feature_dir, 'w') as f:
                f.write('timeout')
            with open(time_dir, 'w') as f:
                f.write('timeout')


def Projection(polys, first_var, other_vars, config, projected):
    vars_dir = projected + first_var
    
    maple_dir = f'datasets/dataset_{config.mode}/projection/mpl/' + config.file + '_' + vars_dir + '.mpl'
    data_dir = f'datasets/dataset_{config.mode}/projection/txt/' + config.file + '_' + vars_dir + '.txt'
 
    if not os.path.exists(data_dir):
        with open(maple_dir, 'w') as f:
            f.write('with(ProjectionCAD_utils):\n')
            f.write('polys_ori:={' + polys + '}:\n')
            f.write(f'projs:=PCAD_McCallumProj(polys_ori,{first_var},[{other_vars}]):\n')
            f.write('projs_fac:=PCAD_SetFactors(projs):\n')
            f.write('projs_fac:=map(x->expand(denom(x) * x), projs_fac):\n')
            f.write(f'writestat(\"{data_dir}\", projs_fac):\n')
            f.write('quit;\n')
        
        cmd = 'maple ' + maple_dir
        p = subprocess.Popen(cmd, shell=True)
        p.wait()

        # kill_mserver = 'ps -ef | grep ' + maple_dir + ' | grep -v grep | awk \'{print $2}\' | xargs kill -s 9'
        # p1 = subprocess.Popen(kill_mserver, shell=True)
        # p1.wait()


def proj_and_feature(polys, first_var, other_vars, config, projected):
    vars_dir = projected + first_var
    
    maple_dir = 'datasets/' + config.dataset + '/projection/mpl/' + config.file + '_' + vars_dir + '.mpl'
    data_dir = 'datasets/' + config.dataset + '/projection/txt/' + config.file + '_' + vars_dir + '.txt'
    time_dir = 'datasets/' + config.dataset + '/projection/time/' + config.file + '_' + vars_dir + '.time'
    feature_dir = 'datasets/' + config.dataset + '/projection/txt/' + config.file + '_' + vars_dir + '.f'
 
    if not os.path.exists(data_dir):
        with open(maple_dir, 'w') as f:
            f.write('with(ProjectionCAD_utils):\n')
            f.write('polys_ori:={' + polys + '}:\n')
            f.write("kernelopts(printbytes=false):\n")
            f.write(f'projs:=PCAD_McCallumProj(polys_ori,{first_var},[{other_vars}]):\n')
            f.write('projs_fac:=PCAD_SetFactors(projs):\n')
            f.write('vars_fac:=indets(projs_fac):\n')
            f.write('projs_fac:=map(x->expand(denom(x) * x), projs_fac):\n')
            f.write('features:=get_features(convert(projs_fac, list), convert(vars_fac, list)):\n')
            f.write(f'writestat(\"{data_dir}\", [projs_fac, vars_fac]):\n')
            f.write(f'writestat(\"{feature_dir}\", features):\n')
            f.write('quit;\n')
        
        cmd = 'maple ' + maple_dir

        try:
            projection_time = time.time()
            p = subprocess.Popen(cmd, shell=True)
            p.wait(timeout=(config.remain_time))
            projection_time = time.time() - projection_time
            with open(time_dir, 'w') as f:
                f.write(str(projection_time))

            if os.path.exists(data_dir):
                with open(data_dir) as f:
                    line = f.readline().strip()
                if line[:10] == '[projs_fac':
                    with open(data_dir, 'w') as f:
                        f.write('error')
            else:
                with open(data_dir, 'w') as f:
                    f.write('error')

            if os.path.exists(feature_dir):
                with open(feature_dir) as f:
                    line = f.readline().strip()
                if line == 'features':
                    with open(feature_dir, 'w') as f:
                        f.write('error')
            else:
                with open(feature_dir, 'w') as f:
                    f.write('error')

            # kill_mserver = 'ps -ef | grep ' + maple_dir + ' | grep -v grep | awk \'{print $2}\' | xargs kill -s 9'
            # p1 = subprocess.Popen(kill_mserver, shell=True)
            # p1.wait()

        except subprocess.TimeoutExpired:
            kill_mserver = 'ps -ef | grep ' + maple_dir + ' | grep -v grep | awk \'{print $2}\' | xargs kill -s 9'
            p1 = subprocess.Popen(kill_mserver, shell=True)
            p1.wait()

            with open(data_dir, 'w') as f:
                f.write('timeout')
            with open(feature_dir, 'w') as f:
                f.write('timeout')
            with open(time_dir, 'w') as f:
                f.write('timeout')
