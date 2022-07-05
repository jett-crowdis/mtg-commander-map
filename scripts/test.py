import glob
import logging
    

def check_submap_structure(folder):
    '''Checks the data structure of an output folder'''

    # check that every submap has required files
    for submap in glob.glob(folder + '/submaps/*/*'):
        submap_files = [f.split('/')[-1] for f in glob.glob(submap + '/*')]
        submap_name = '/'.join(submap.split('/')[-2:])
        for f in ['edh-submap.csv', 'edh-submap-clusters.json', 'submap.json']:
            if f not in submap_files:
                logging.debug(f'submap {submap_name} is missing {f}')

                
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Runs tests on data from the Commander Map'
    )

    parser.add_argument('--data_dir', type=str,
                        help='directory containing Commander Map data')

    args = parser.parse_args()
    data_dir = args.data_dir
    
    # check that every submap has required files
    logging.info('\n###########################\nChecking submap directories\n###########################\n')
    check_submap_structure(data_dir)
    
    