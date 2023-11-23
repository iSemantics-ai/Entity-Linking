FROM python:3.10
# Copy the requirements.txt file to the container
COPY cpu_requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r cpu_requirements.txt


# create workdir
RUN mkdir entity_linking


# Copy the contents of the code directory to the container
COPY /matcher /entity_linking/matcher
COPY inference_script.py /entity_linking

# Set the working directory in the container
WORKDIR /entity_linking
# # Set environment variable to enable GPU usage
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Define the command to run your Python file
# CMD ["python", "inference_script.py"]
