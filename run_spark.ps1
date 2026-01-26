$env:JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-11.0.30.7-hotspot"

$env:PATH="C:\Program Files\Eclipse Adoptium\jdk-11.0.30.7-hotspot\bin;$env:PATH"

Write-Host "JAVA_HOME =" $env:JAVA_HOME

java -version

python event_driven_pipeline/spark_pipeline_trigger_job.py