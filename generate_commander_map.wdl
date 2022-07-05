workflow commander_map {
    String edhrec_url
    String date
    File aws_credentials
    File edhrec_to_scryfall
    File slug_image_override
    File duplicates
    String additional_flags

    call preprocess_data {
        input:
            edhrec_url=edhrec_url,
            date=date,
            edhrec_to_scryfall=edhrec_to_scryfall,
            slug_image_override=slug_image_override,
            duplicates=duplicates,
            additional_flags=additional_flags
    }

    call generate_main_map_coordinates {
        input:
            preprocessed_data=preprocess_data.preprocessed_data
    }

    call generate_main_map_clusters {
        input:
            preprocessed_data=preprocess_data.preprocessed_data,
            additional_flags=additional_flags
    }

    scatter (idx in preprocess_data.indices) {
        call generate_submaps {
            input:
                index=idx,
                submap_file=preprocess_data.submaps[idx],
                preprocessed_data=preprocess_data.preprocessed_data,
                date=date,
                additional_flags=additional_flags
        }
    }

    call merge_map_data {
        input:
            coordinates=generate_main_map_coordinates.map_coordinates,
            clusters=generate_main_map_clusters.map_clusters,
            submap_array=generate_submaps.submaps,
            map_cluster_data=generate_main_map_clusters.map_cluster_data,
            map_cluster_folder=generate_main_map_clusters.cluster_folder,
            date=date,
            aws_credentials=aws_credentials,
            decks=preprocess_data.decks
    }

    output {
        File raw_data=preprocess_data.raw_data
        File map_data=merge_map_data.map_data
    }
}

task preprocess_data {
    String edhrec_url
    String date
    File edhrec_to_scryfall
    File slug_image_override
    File duplicates
    String additional_flags
    Int? n_split = 36

    String edhrec_date = basename(edhrec_url)

    Int? memoryGB = 16
    Int? diskGB = 64
    Int? preemptible_attempts = 3

    command <<<

        # fail nonsilently
        set -euxo pipefail

        # get the EDHREC data
        wget ${edhrec_url}
        mkdir data
        tar -zxf ${edhrec_date} -C data
        
        # preprocess the data - clean it, break submaps into parallelized groups, and calculate submap traits
        python -u /scripts/preprocess_data.py --data_dir data \
                                              --edhrec_to_scryfall ${edhrec_to_scryfall} \
                                              --out_dir ${date} \
                                              --duplicates ${duplicates} \
                                              --slug_image_override ${slug_image_override} \
                                              --n_split ${n_split} \
                                              ${additional_flags}

        # tar files for export
        tar -czf data.tar.gz data
        tar -czf ${date}.tar.gz ${date}
        tar -czf decks.tar.gz decks
    >>>

    runtime {
        docker: "jcrowdis/commander-map:2.3"
        memory: "${memoryGB} GB"
        disks: "local-disk ${diskGB} SSD"
        preemptible: preemptible_attempts
    }

    output {
        File raw_data="data.tar.gz"
        File preprocessed_data="${date}.tar.gz"
        File decks="decks.tar.gz"
        Array[File] submaps=glob("grouped_data_*.csv")
        Array[Int] indices=read_lines("indices.txt")
    }
}

task generate_main_map_coordinates {
    File preprocessed_data
    Int? n_neighbors = 25

    Int? memoryGB = 64
    Int? diskGB = 64
    Int? preemptible_attempts = 3

    command <<<

        # fail nonsilently
        set -euxo pipefail

    	mkdir preprocessed_data
        tar -zxf ${preprocessed_data} --strip-components=1 -C preprocessed_data
        
        # generate the coordinates of the main map using UMAP with n_neighbors (min_dist = 0.10)
        python -u /scripts/generate_main_map_coordinates.py --preprocessed_data preprocessed_data \
                                                            --n_neighbors ${n_neighbors}
    >>>

    runtime {
        docker: "jcrowdis/commander-map:2.3"
        memory: "${memoryGB} GB"
        disks: "local-disk ${diskGB} SSD"
        preemptible: preemptible_attempts
    }

    output {
        File map_coordinates="commander-map-coordinates.csv"
    }
}

task generate_main_map_clusters {
    File preprocessed_data
    String additional_flags

    Int? memoryGB = 64
    Int? diskGB = 64
    Int? preemptible_attempts = 3

    command <<<

        # fail nonsilently
        set -euxo pipefail

        mkdir preprocessed_data
        tar -zxf ${preprocessed_data} --strip-components=1 -C preprocessed_data
        
        # cluster the main map using UMAP and HDBSCAN
        python -u /scripts/generate_main_map_clusters.py --preprocessed_data preprocessed_data \
                                                         ${additional_flags}

        # compress for output
        tar -czf clusters.tar.gz clusters
    >>>

    runtime {
        docker: "jcrowdis/commander-map:2.3"
        memory: "${memoryGB} GB"
        disks: "local-disk ${diskGB} SSD"
        preemptible: preemptible_attempts
    }

    output {
        File map_embedding="map-embedding.csv"
        File map_clusters="commander-map-clusters.csv"
        File map_cluster_data="edh-map-clusters.json"
        File cluster_folder="clusters.tar.gz"
    }
}

task generate_submaps {
    Int index
    File submap_file
    File preprocessed_data
    String date
    String additional_flags

    Int? memoryGB = 24
    Int? diskGB = 64
    Int? preemptible_attempts = 3

    command <<<

        # fail nonsilently
        set -euxo pipefail

    	mkdir preprocessed_data
        tar -zxf ${preprocessed_data} --strip-components=1 -C preprocessed_data

        # generate all the submaps within submap_file
        python -u /scripts/generate_submaps.py --preprocessed_data preprocessed_data \
                                               --submap_file ${submap_file} \
                                               ${additional_flags}

        # compress for output
        tar -czf ${date}_${index}.tar.gz preprocessed_data
    >>>

    runtime {
        docker: "jcrowdis/commander-map:2.3"
        memory: "${memoryGB} GB"
        disks: "local-disk ${diskGB} SSD"
        preemptible: preemptible_attempts
    }

    output {
        File submaps="${date}_${index}.tar.gz"
    }
}

task merge_map_data {
    File coordinates
    File clusters
    Array[File] submap_array
    String date
    File map_cluster_data
    File map_cluster_folder
    File aws_credentials
    File decks

    String dollar = "$"

    Int? memoryGB = 8
    Int? diskGB = 64
    Int? preemptible_attempts = 3

    command <<<

    # fail nonsilently
    set -euxo pipefail

    # we start by combining all the submap data
    submaps='${sep=" " submap_array}'
    for sbmp in ${dollar}{submaps}; do
        echo "unzipping $sbmp"
        tar -zxf $sbmp
    done

    mv preprocessed_data ${date}

    # move the decks into this folder
    tar -zxf ${decks} -C ${date}

    # merge map data and move it
    awk -F "," '{print $NF}' ${clusters} > clusters.csv
    paste -d , ${coordinates} clusters.csv > ${date}/edh-map.csv
    
    # move the cluster data into our output
    mv ${map_cluster_data} ${date}/map_intermediates/edh-map-clusters.json
    tar -zxf ${map_cluster_folder} -C ${date}

    # upload to aws. A disgusting hack to get around WDL's reserved usage of $
    readarray -t credentials < ${aws_credentials}
    AWS_ACCESS_KEY_ID=${dollar}{credentials[0]} AWS_SECRET_ACCESS_KEY=${dollar}{credentials[1]} \
        aws s3 cp --recursive ${date}/ s3://data.luckypaper.co/commander-map/${date} \
        --acl public-read --content-type "text/plain" --quiet

    # compress for output
    tar -czf ${date}.tar.gz ${date}
    >>>

    runtime {
        docker: "jcrowdis/commander-map:2.3"
        memory: "${memoryGB} GB"
        disks: "local-disk ${diskGB} SSD"
        preemptible: preemptible_attempts
    }

    output {
        File map_data="${date}.tar.gz"
    }
}