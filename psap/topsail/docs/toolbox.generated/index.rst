
Toolbox Documentation
=====================
            

``configure``
*************

::

    Commands relating to TOPSAIL testing configuration
    

                
* :doc:`apply <Configure.apply>`	 Applies a preset (or a list of presets) to the current configuration file
* :doc:`enter <Configure.enter>`	 Enter into a custom configuration file for a TOPSAIL project
* :doc:`get <Configure.get>`	 Gives the value of a given key, in the current configuration file
* :doc:`name <Configure.name>`	 Gives the name of the current configuration

``run``
*******

::

    Run `topsail` toolbox commands from a single config file.
    

                

``plotter``
***********

::

    Commands related to the current role
    

                
* :doc:`main <Plotter.main>`	 Run the plotter role

``repo``
********

::

    Commands to perform consistency validations on this repo itself
    

                
* :doc:`generate_ansible_default_settings <Repo.generate_ansible_default_settings>`	 Generate the `defaults/main/config.yml` file of the Ansible roles, based on the Python definition.
* :doc:`generate_middleware_ci_secret_boilerplate <Repo.generate_middleware_ci_secret_boilerplate>`	 Generate the boilerplace code to include a new secret in the Middleware CI configuration
* :doc:`generate_toolbox_related_files <Repo.generate_toolbox_related_files>`	 Generate the rst document and Ansible default settings, based on the Toolbox Python definition.
* :doc:`generate_toolbox_rst_documentation <Repo.generate_toolbox_rst_documentation>`	 Generate the `doc/toolbox.generated/*.rst` file, based on the Toolbox Python definition.
* :doc:`send_job_completion_notification <Repo.send_job_completion_notification>`	 Send a *job completion* notification to github and/or slack about the completion of a test job.
* :doc:`validate_no_broken_link <Repo.validate_no_broken_link>`	 Ensure that all the symlinks point to a file
* :doc:`validate_no_wip <Repo.validate_no_wip>`	 Ensures that none of the commits have the WIP flag in their message title.
* :doc:`validate_role_files <Repo.validate_role_files>`	 Ensures that all the Ansible variables defining a filepath (`project/*/toolbox/`) do point to an existing file.
* :doc:`validate_role_vars_used <Repo.validate_role_vars_used>`	 Ensure that all the Ansible variables defined are actually used in their role (with an exception for symlinks)

``smigather``
*************

::

    Commands related to the current role
    

                
* :doc:`main <Smigather.main>`	 Run the Smigather role

``tests``
*********

::

    Commands related to the current role
    

                
* :doc:`main <Tests.main>`	 Run the tests role

``whisper``
***********

::

    Commands related to the current role
    

                
* :doc:`main <Whisper.main>`	 Run the whisper role
