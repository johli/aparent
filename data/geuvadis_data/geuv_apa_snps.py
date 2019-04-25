import subprocess
import shlex


'''

/samtools/bcftools$ ./bcftools query -f '%TYPE\t%CHROM\t%POS\t%REF\t%ALT[\t%SAMPLE=%GT]\n' --output ../snp1:52253461-52254084.txt --regions 1:52253461-52254084 ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase1/analysis_results/integrated_call_sets/ALL.chr1.integrated_phase1_v3.20101123.snps_indels_svs.genotypes.vcf.gz

query -f \'%TYPE\t%CHROM\t%POS\t%REF\t%ALT[\t%SAMPLE=%GT]\n\' 

chr8:90945710,90947808	-	8:90947607-90947756	ATGCAATGACAAAGCCTGAAAACAGAACAAACAATTGTTACATACAAAAGAATCAAAGTTTTGTGCATTTTATTTAATAAATTTAGGCCATAAAACATTGTAACTTAAATCGCTTCTATACACTATATATTCATATAACCTTGTTGGCCT	150

'''

g1000_base_start = 'ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase1/analysis_results/integrated_call_sets/ALL.'
g1000_base_end = '.integrated_phase1_v3.20101123.snps_indels_svs.genotypes.vcf.gz'


print('Matching APA events against 1000 Genomes Genotype data.')
with open('Emitted_Events_APA_100up_50dn.txt') as f:
	i = 0
	for line in f:

		print('Processing event: ' + str(i))
		print(line)

		lineparts = line.split('\t')
		gene_key = lineparts[0]
		strand = lineparts[1]
		search_region = lineparts[2]
		sequence = lineparts[3]

		output_file = './snps/' + gene_key + '_' + search_region + '.txt'
		chrom = gene_key.split(':')[0]

		genotype_file = g1000_base_start + chrom + g1000_base_end

		subprocess.call(shlex.split('../samtools/bcftools/bcftools query -f \'%TYPE\t%CHROM\t%POS\t%REF\t%ALT[\t%SAMPLE=%GT]\n\' --output ' + output_file + ' --regions ' + search_region + ' ' + genotype_file))

		#subprocess.call(shlex.split('../samtools/bcftools/bcftools query -f \'%TYPE\t%CHROM\t%POS\t%REF\t%ALT[\t%SAMPLE=%GT]\n\' --output ./snp1:52253461-52254084.txt --regions 1:52253461-52254084 ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase1/analysis_results/integrated_call_sets/ALL.chr1.integrated_phase1_v3.20101123.snps_indels_svs.genotypes.vcf.gz'))

		i += 1




