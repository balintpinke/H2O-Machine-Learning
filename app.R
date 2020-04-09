# setwd("D:/R_WD/H2O-Machine Learning-Shiny")

library(shiny)
library(shinydashboard)
library(DT)
library(h2o)

#start up 1-node H2O server on local machines(all CPU cores and 2GB memory)
h2o.init(nthreads=-1, max_mem_size="2G")

# H2OFrame -----
df <- h2o.importFile(path = normalizePath("D:/R_WD/H2O-Machine Learning-Shiny/NHL_2016_17.csv"))
df_dataframe = as.data.frame(df)
# training, test datasets -----

splits <- h2o.splitFrame(df, 0.75, seed=666)
train  <- h2o.assign(splits[[1]], "train_df") # 75%
test   <- h2o.assign(splits[[2]], "test_df")  # 25%
rm(splits)

response_var <- "Position_cat4"
predictor_vars <- names(train)[7:140]


ui <- dashboardPage(
       dashboardHeader(title="Machine learning in H2O - Predicting positions", titleWidth = 450),
        dashboardSidebar(
         sidebarMenu(
          menuItem("The Data", tabName = "nhl_data", icon = icon("hockey-puck")),
          menuItem("Prediction", tabName = "prediction", icon = icon("user-friends"))
          )),
  dashboardBody(
    tabItems(
      # 1. tab content
      tabItem("nhl_data",
          fluidRow(
            box(title = "NHL season 2016/17 Data", width = 12, solidHeader = TRUE, status = "primary",
              dataTableOutput("NHL_data")
          )
      )),
      tabItem("prediction",
          fluidRow(
          actionButton("start_ml", "Start prediction"),
      box(title = "Neural net", width = 12, solidHeader = TRUE, status = "primary",
        tableOutput("confusion_matrix1")
        ),
      box(title = "Random forest", width = 12, solidHeader = TRUE, status = "primary",
        tableOutput("confusion_matrix2")
        ),
      box(title = "Gradient boosted machines", width = 12, solidHeader = TRUE, status = "primary",
        tableOutput("confusion_matrix3")
        ),
      box(title = "Generalized linear model", width = 12, solidHeader = TRUE, status = "primary",
        tableOutput("confusion_matrix4")
       )
  )
  )
  )
)
)

server <- function(input, output) { 
  
  output$NHL_data <- renderDataTable({
    datatable(df_dataframe, 
    options = list(
      scrollX = TRUE)
    )
    })
  
  
  results_lists <- eventReactive(input$start_ml, {
    
  nn_model <- h2o.deeplearning(
    model_id="neuralnet_classification",  
    training_frame=train,
    x=predictor_vars,
    y=response_var,
    standardize=TRUE,
    nfolds=5,
    fold_assignment = "Stratified",
    balance_classes = TRUE,
    shuffle_training_data = TRUE,
    activation="Rectifier",## default
    hidden=c(32, 32),  ## default: 2 hidden layers with 200 neurons each
    epochs=10,      #How many times the dataset should be iterated
    adaptive_rate = TRUE)
  
  
  rf_modell <- h2o.randomForest(
    model_id="randomforest_classification",
    training_frame = train,
    x=predictor_vars,
    y=response_var,
    balance_classes = TRUE,
    fold_assignment = "Stratified",
    nfolds = 5,
    ntrees = 50, ## default=50
    max_depth = 20, # default=20
    # mtries = c(9, 14, 19), #default p/3
    sample_rate = 0.632, # default
    min_rows = 1 #Fewest allowed observations in a leaf (default = 1)
  )
  
  
  gbm_modell <- h2o.gbm(
    model_id="gradient_boosted_machines_classification",
    training_frame = train,
    x=predictor_vars,
    y=response_var,
    balance_classes = TRUE,
    fold_assignment = "Stratified",
    nfolds = 5,
    min_rows = 10 #default = 10
  )
  
  glm_modell <- h2o.glm(
    model_id="generalized_linear_models_classification",
    training_frame = train,
    x=predictor_vars,
    y=response_var,
    standardize = TRUE,
    fold_assignment = "Stratified",
    nfolds = 5,
    family = "multinomial")
  
  x1=as.data.frame(h2o.confusionMatrix(nn_model, newdata=test))
  x2=as.data.frame(h2o.confusionMatrix(rf_modell, newdata=test))
  x3=as.data.frame(h2o.confusionMatrix(gbm_modell, newdata=test))
  x4=as.data.frame(h2o.confusionMatrix(glm_modell, newdata=test))
  
  
  list=list(x1=x1, x2=x2, x3=x3, x4=x4)
  list
  })
  
  output$confusion_matrix1 <- renderTable(results_lists()[["x1"]])
  
  output$confusion_matrix2 <- renderTable(results_lists()[["x2"]])
  
  output$confusion_matrix3 <- renderTable(results_lists()[["x3"]])
  
  output$confusion_matrix4 <- renderTable(results_lists()[["x4"]])
  
  

  
  

}
  
shinyApp(ui, server)











