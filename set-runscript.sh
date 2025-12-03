az webapp config set \
    --startup-file "gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8080 agent:app" \
    --name $APP_SERVICE_NAME \
    --resource-group $RESOURCE_GROUP_NAME