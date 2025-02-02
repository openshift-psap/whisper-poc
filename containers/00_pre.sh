oc new-project my-whisper-runtime
oc adm policy add-scc-to-user privileged -z default -n my-whisper-runtime
