 FROM python:3.11.5-slim
    
     # Set the working directory
     WORKDIR /app
    
    # Copy requirements and install packages
      COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
   
    # Copy all your project code
   COPY . .
   
    # Expose the port your application listens on
    EXPOSE 5000
   
    # Run the main application directly
    CMD ["python", "app/app.py"]