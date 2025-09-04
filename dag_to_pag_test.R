if (!requireNamespace("this.path", quietly = TRUE)) {
  install.packages("this.path")
}
library(this.path)
setwd(this.dir())
library(pcalg)
library(graph)
library(jsonlite)
####Part one: DAG to PAG####
DAGtoPAG <- function(graph_df,verbose = FALSE){
  
  dag <- graph_df
  dag.matrix <- as.matrix(dag)
  dag.nodes.name <- colnames(dag)
  
  edL <- vector("list", length=nrow(dag.matrix))
  names(edL) <- dag.nodes.name
  
  for (i in seq_len(nrow(dag.matrix))) {
    ch <- which(dag.matrix[i, ] == 1)
    ch <- as.numeric(ch)
    if (length(ch) > 0) {
      edL[[i]] <- list(edges=dag.nodes.name[ch])
    }
    else
    {
      edL[[i]] <- list(NULL)
    }
  }
  
  true.dag <- graphNEL(nodes=dag.nodes.name, edgeL=edL,"directed")
  L.names <- dag.nodes.name[grep("^L", dag.nodes.name)]
  cat("dag.nodes.name: ", dag.nodes.name, "\n")
  cat("Latent nodes: ", L.names, "\n")
  L <- which(dag.nodes.name %in% L.names)
  L <- as.numeric(L)
  
  cov.mat <- trueCov(true.dag)
  true.corr <- cov2cor(cov.mat)
  rules <-  rep(TRUE,10)
  off <- c(5,6,7) # no selecion bias
  rules[off] <- FALSE
  true.pag <- dag2pag(suffStat = list(C = true.corr, n = 10^9),
                                indepTest = gaussCItest,
                                graph=true.dag,
                                L=L, alpha = 0.9999, rules = rules, verbose = verbose)
  pag.nodes_name <- dag.nodes.name[!dag.nodes.name %in% L.names]
  cat("pag.nodes_name: ", pag.nodes_name, "\n")
  # Replace the row and column labels of true.pag@amat with pag.nodes_name
  rownames(true.pag@amat) <- pag.nodes_name
  colnames(true.pag@amat) <- pag.nodes_name
  pag.matrix <- true.pag@amat
  
  pag.matrix
  
  
}


# Load DAGs from JSON file
num.nodes <- 30
average.degree <- 3
path <- paste0("Test_Data/dags_", num.nodes, "_", average.degree, ".json")

dags <- fromJSON(path, simplifyVector = FALSE)
pags <- vector("list", length = length(dags))
pag.matrices <- vector("list", length = length(dags))

for (i in seq_along(dags)) {
  graph_info <- dags[[i]]
  graph_id <- graph_info$id
  graph_csv <- graph_info$graph
  graph_df <- read.csv(text = graph_csv, row.names = 1)
  cat("Loaded graph:", graph_id, "\n")
  pag_matrix <- DAGtoPAG(graph_df, verbose = FALSE)
  pag.matrices[[i]] <- pag_matrix
  pag_csv <- paste(capture.output(write.csv(pag_matrix, row.names = TRUE)), collapse = "\n")
  pags[[i]] <- list(
    id = graph_id,
    graph = pag_csv
  )
}

# Save PAGs to JSON file
path_pag <- paste0("Test_Data/pcalg_pags_", num.nodes, "_", average.degree, ".json")
write_json(pags, path_pag, pretty = TRUE, auto_unbox = TRUE)

