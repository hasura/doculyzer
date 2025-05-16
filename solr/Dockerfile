FROM solr:9.3

# Switch to root for file operations
USER root

# Copy init script
COPY init-cores.sh /opt/docker-solr/scripts/
RUN chmod +x /opt/docker-solr/scripts/init-cores.sh

# Ensure proper permissions
RUN chown solr:solr /opt/docker-solr/scripts/init-cores.sh

# Environment variables
ENV SOLR_HEAP=1024m
ENV SOLR_JAVA_MEM="-Xms512m -Xmx1024m"

# Switch back to solr user
USER solr

# Set the script as the entrypoint
CMD ["/opt/docker-solr/scripts/init-cores.sh"]
