envvars:
    "SPECFEMPP_BINDIR",


pathvars:
    cwd=os.getcwd()


rule meshfem:
    input:
        provenance="<cwd>/provenance",
    output:
        "<cwd>/database.bin",
    shell:
        """
        cd {input.provenance}
        $SPECFEMPP_BINDIR/xmeshfem2D -p Par_file
        """

rule clean:
    shell:
        """
        rm -f database.bin
        rm -rf provenance/OUTPUT_FILES
        """
