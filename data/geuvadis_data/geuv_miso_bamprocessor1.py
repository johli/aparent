import subprocess
import shlex
import os.path
import time


'''

ONE TIME:

index_gff --index TandemUTR.hg19.gff3 indexed/

exon_utils --get-const-exons Homo_sapiens.GRCh37.65.gff --min-exon-size 1000 --output-dir exons/


PER SAMPLE:

wget -O /home/johli/apa_snp/geuvadis/HG00096.1.M_111124_6.bam "http://www.ebi.ac.uk/arrayexpress/files/E-GEUV-1/HG00096.1.M_111124_6.bam"

samtools sort /home/johli/apa_snp/geuvadis/HG00096.1.M_111124_6.bam /home/johli/apa_snp/geuvadis/HG00096.1.M_111124_6.sorted

samtools index /home/johli/apa_snp/geuvadis/HG00096.1.M_111124_6.sorted.bam

pe_utils --compute-insert-len geuvadis/HG00096.1.M_111124_6.sorted.bam exons/Homo_sapiens.GRCh37.65.with_chr.min_1000.const_exons.gff --output-dir insert-dist/

HG00096
mean	sdev	dispersion
194.7	47.4	3.4

miso --run indexed geuvadis/HG00096.1.M_111124_6.sorted.bam --output-dir geuvadis_output/ --read-len 75 --paired-end 194 47

summarize_miso --summarize-samples geuvadis_output/ geuvadis/HG00096_summary/

rm geuvadis/HG00096.1.M_111124_6.bam
rm geuvadis/HG00096.1.M_111124_6.sorted.bam
rm geuvadis/HG00096.1.M_111124_6.sorted.bam.bai

rm insert-dist/bam2gff_Homo_sapiens.GRCh37.65.with_chr.min_1000.const_exons.gff/HG00096.1.M_111124_6.sorted.bam

rm -rf geuvadis_output

rm -rf ~/.local/share/Trash/*

'''

sample_ftp = {}
sample_bam = {}

with open('geuvadis/E-GEUV-1.sdrf.txt') as f:
	i = 0
	for line in f:

		if i > 0 :
			line_parts = line.split('\t')

			print(line_parts[0])
			print(line_parts[len(line_parts) - 4])
			print(line_parts[len(line_parts) - 3])

			sample_bam[line_parts[0]] = line_parts[len(line_parts) - 4]
			sample_ftp[line_parts[0]] = line_parts[len(line_parts) - 3]

		i += 1


print('Compiling Miso Tandem UTR Summaries for ' + str(len(sample_bam)) + ' samples.')

samples = []
for sample in sample_bam :
	samples.append(sample)

samples.sort()

for sample in samples :
	if sample == 'HG00096' :
		continue

	if os.path.exists('/home/johli/apa_snp/geuvadis/' + sample + '_summary') :
		continue
	if os.path.exists('/home/johli/apa_snp/geuvadis/' + sample + '_bamprocessor1_complete.txt') :
		continue

	print('Sample: ' + sample)

	ftp_address = sample_ftp[sample]
	bam_file = sample_bam[sample]
	file_name = bam_file[:len(bam_file) - 4]

	while not os.path.exists('/home/johli/apa_snp/geuvadis/' + sample + '_bamindexer_complete.txt') :
		time.sleep(5)

	subprocess.call(shlex.split('pe_utils --compute-insert-len geuvadis/' + file_name + '.sorted.bam exons/hamlet.gff --output-dir insert-dist/'))	

	subprocess.call(shlex.split('touch /home/johli/apa_snp/geuvadis/' + sample + '_bamprocessor1_complete.txt'))