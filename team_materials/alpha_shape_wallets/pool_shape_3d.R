# Rebuild the alpha shape of all wallets (k = 2.74) and the MST-dropped sets at other k. Run from alpha_shape_wallets/. ~20 min.
source("wallet_alpha_analysis.R")
ar <- skip_scan_display(use_fast_mst(load_alpharichness(CONFIG$pkg_dir)))
w <- read.csv(CONFIG$wallet_file, colClasses=c(wallet="character")); w <- w[CONFIG$pool_filter(w),]
tr <- build_traits(w, CONFIG$trait_fn, 6)
t0 <- Sys.time()
res <- ar$alpha_richness(tr, k = 2.74)
faces <- ar$boundary_faces(res$simp_included)
saveRDS(list(pts = res$pts, faces = faces, cut_count = res$cut_count, volume = res$volume, k = 2.74,
             in_shape = seq_len(nrow(res$pts)) %in% res$simp_included, mst_dropped = res$point_cluster == 0),
        "alpha_results_pca3d/pool_shape_3d.rds")
message("cut ", res$cut_count, " faces ", nrow(faces), " time ", format(Sys.time() - t0))
# robustness: MST-dropped set at several k (cheap: MST built once)
mst <- ar$mst_edges(res$pts); q <- quantile(mst$len, c(.25,.75))
kg <- c(1.5, 2, 2.5, 2.74, 3)
drop <- sapply(kg, function(k){ cut <- q[[2]] + k*(q[[2]]-q[[1]]); g <- mst$len > cut
  comp <- ar$edge_components(nrow(res$pts), mst$from[!g], mst$to[!g]); size <- tabulate(comp); size[comp] < 4 })
colnames(drop) <- paste0("k_", kg); rownames(drop) <- rownames(res$pts)
saveRDS(drop, "alpha_results_pca3d/mst_drop_by_k.rds"); message("done")
