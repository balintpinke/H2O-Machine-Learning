# setwd("D:/R_WD/H2O-Machine Learning-Shiny")

library(shiny)
library(shinydashboard)
library(DT)
library(shinyWidgets)
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
         sidebarMenu(id="menu",
          menuItem("The Data", tabName = "nhl_data", icon = icon("hockey-puck")),
          menuItem("Prediction", tabName = "prediction", icon = icon("bullseye")),
          conditionalPanel(
            condition = "input.menu == 'prediction'",
            pickerInput(
              inputId = "predictors",
              label = "Choose predictor variables to predict player's position",
              choices = names(df_dataframe)[5:140],
              multiple = TRUE,
              options = list(
                `actions-box` = TRUE,
                `deselect-all-text` = "None",
                `select-all-text` = "All variable",
                `none-selected-text` = "Select variables"
              )
          )
          )
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
        box(title = "Add input parameters and run algorithms", width = 12, solidHeader = TRUE, status = "primary",
        column(width = 6,
      box(title = "Neural net", width = 12, solidHeader = TRUE, status = "primary",
          
          numericInput(inputId = "nfolds_nn", label = "Number of K-Fold Cross-validation set", 
                       min = 3, value = 5, max = 10),
          numericInput(inputId = "epoch", label = "Number of epochs", 
                       min = 1, value = 10, max = 100),
          textInput('hidden_layer', 'Add hidden layers of neurons (2 layer with 32 neurons each', "32,32"),
          actionButton("start_nn", "Run neural network", icon = icon("fas fa-play-circle"))
        )
      ),
        column(width = 6,
      box(title = "Random forest", width = 12, solidHeader = TRUE, status = "primary",
          
          numericInput(inputId = "nfolds_rf", label = "Number of K-Fold Cross-validation set", 
                       min = 3, value = 5, max = 10),
          numericInput(inputId = "ntrees_rf", label = "Number of trees in the forest", 
                       min = 10, value = 50, max = 150),
          numericInput(inputId = "maxdepth_rf", label = "How deep a tree is allowed to grow", 
                       min = 5, value = 20, max = 100),
          actionButton("start_rf", "Run random forest", icon = icon("fas fa-play-circle"))
        )
      ),
        column(width = 6,
      box(title = "Gradient boosted machines", width = 12, solidHeader = TRUE, status = "primary",
          
          numericInput(inputId = "nfolds_gbm", label = "Number of K-Fold Cross-validation set", 
                       min = 3, value = 5, max = 10),
          numericInput(inputId = "ntrees_gbm", label = "Number of trees in the forest", 
                       min = 1, value = 5, max = 50),
          numericInput(inputId = "maxdepth_gbm", label = "How deep a tree is allowed to grow", 
                       min = 5, value = 20, max = 100),
          actionButton("start_gbm", "Run gradient boosted machines", icon = icon("fas fa-play-circle"))
        )
      ),
        column(width = 6,
      box(title = "Generalized linear model", width = 12, solidHeader = TRUE, status = "primary",
          numericInput(inputId = "nfolds_glm", label = "Number of K-Fold Cross-validation set", 
                       min = 3, value = 5, max = 10),
          numericInput(inputId = "alpha", label = "alpha - L1 reguralization, (1-alpha - L2 reguralization", 
                       min = 0, value = 0.5, max = 0),
          actionButton("start_glm", "Run multinomial generalized linear model", icon = icon("fas fa-play-circle"))
          
        
      )
    )
  ),
  
  box(title = "Model results - Confusion matrices", width = 12, solidHeader = TRUE, status = "primary",
      column(width = 6,
      box(title = "Neural net", width = 12, solidHeader = TRUE, status = "primary",
      tableOutput("confusion_matrix1")
      )
      ),
      column(width = 6,
      box(title = "Random forest", width = 12, solidHeader = TRUE, status = "primary",
      tableOutput("confusion_matrix2")
      )
      ),
      column(width = 6,
      box(title = "Gradient boosted machines", width = 12, solidHeader = TRUE, status = "primary",
      tableOutput("confusion_matrix3")
      )
      ),
      column(width = 6,
      box(title = "Generalized linear model", width = 12, solidHeader = TRUE, status = "primary",
      tableOutput("confusion_matrix4")
      )
      )
  ) #Model results box
          )
      ) #tabItem
  ) #tabItems
)
)

server <- function(input, output) { 
  
  output$NHL_data <- renderDataTable({
    datatable(df_dataframe, 
    options = list(
      scrollX = TRUE)
    )
    })
  
  
nn_model <- eventReactive(input$start_nn, {
  
  nfolds_number = input$nfolds_nn
  epoch_number = input$epoch
  layers_and_neurons <- as.numeric(unlist(strsplit(input$hidden_layer,",")))
  predictor_vars=input$predictors
  
  nn_model <- h2o.deeplearning(
    model_id="neuralnet_classification",  
    training_frame=train,
    x=predictor_vars,
    y=response_var,
    standardize=TRUE,
    nfolds=nfolds_number,
    fold_assignment = "Stratified",
    balance_classes = TRUE,
    shuffle_training_data = TRUE,
    activation="Rectifier",## default
    hidden=layers_and_neurons,  ## default: 2 hidden layers with 200 neurons each
    epochs=epoch_number,      #How many times the dataset should be iterated
    adaptive_rate = TRUE)
  
  x1=as.data.frame(h2o.confusionMatrix(nn_model, newdata=test))
  x1
  })
  
  
rf_model <- eventReactive(input$start_rf, {
  
  nfolds_number = input$nfolds_rf
  ntrees = input$ntrees_rf
  max_depth = input$maxdepth_rf
  predictor_vars=input$predictors
  
  
  rf_model <- h2o.randomForest(
    model_id="randomforest_classification",
    training_frame = train,
    x=predictor_vars,
    y=response_var,
    balance_classes = TRUE,
    fold_assignment = "Stratified",
    nfolds = nfolds_number,
    ntrees = ntrees, ## default=50
    max_depth = max_depth, # default=20
    # mtries = c(9, 14, 19), #default p/3
    sample_rate = 0.632, # default
    min_rows = 1 #Fewest allowed observations in a leaf (default = 1)
  )
  
  x2=as.data.frame(h2o.confusionMatrix(rf_model, newdata=test))
  x2
  })
  
  
gbm_model <- eventReactive(input$start_gbm, {
  
  nfolds_number = input$nfolds_gbm
  ntrees = input$ntrees_gbm
  max_depth = input$maxdepth_gbm
  predictor_vars=input$predictors
  
  gbm_model <- h2o.gbm(
    model_id="gradient_boosted_machines_classification",
    training_frame = train,
    x=predictor_vars,
    y=response_var,
    balance_classes = TRUE,
    fold_assignment = "Stratified",
    nfolds = nfolds_number,
    ntrees = ntrees, ## default=50
    max_depth = max_depth,
    min_rows = 10 #default = 10
  )
  
  x3=as.data.frame(h2o.confusionMatrix(gbm_model, newdata=test))
  x3
  })
  

glm_model <- eventReactive(input$start_glm, {  
  
  
  nfolds_number = input$nfolds_glm
  predictor_vars=input$predictors
  alpha_number=input$alpha
  
  glm_model <- h2o.glm(
    model_id="generalized_linear_models_classification",
    training_frame = train,
    x=predictor_vars,
    y=response_var,
    standardize = TRUE,
    fold_assignment = "Stratified",
    nfolds = nfolds_number,
    family = "multinomial",
    alpha = alpha_number)
  
  x4=as.data.frame(h2o.confusionMatrix(glm_model, newdata=test))
  x4
  })
  

  
  
  output$confusion_matrix1 <- renderTable(nn_model())
  
  output$confusion_matrix2 <- renderTable(rf_model())
  
  output$confusion_matrix3 <- renderTable(gbm_model())
  
  output$confusion_matrix4 <- renderTable(glm_model())


}
  
shinyApp(ui, server)
