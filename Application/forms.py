from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, StringField
from wtforms.validators import DataRequired

class UploadForm(FlaskForm):
    # folder = FileField('Parent Directory', validators=[DataRequired()])
    folder = StringField('Upload Folder Path:', validators=[DataRequired()], render_kw={"placeholder": "Insert Full Folder Path"})
    submit = SubmitField('Analyze Images')
